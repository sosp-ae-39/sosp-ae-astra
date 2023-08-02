import os
import time
import torch
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, AutoConfig

from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import Counter

from fastertransformer.examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT


class Request:

    def __init__(
        self,
        seq_id: int,
        group_id: int,
        token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> None:
        self.seq_id = seq_id
        self.group_id = group_id
        self.token_ids = token_ids
        self.sampling_params = sampling_params
        self.output_len = sampling_params.max_num_steps
        self.finish = False

        self.prompt_len = len(token_ids)
    
    def get_len(self):
        return self.prompt_len + self.output_len


class RequestGroup:

    def __init__(
        self,
        group_id: int,
        seqs: List[Request],
        arrival_time: float,
    ) -> None:
        self.group_id = group_id
        self.seqs = seqs
        self.arrival_time = arrival_time

    def get_seqs(self) -> List[Request]:
        return self.seqs

    def is_finished(self) -> bool:
        return all(seq.finish for seq in self.seqs)


class FakeFrontend:

    def __init__(
        self,
        # model_name: str,
    ) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq_group_counter = Counter()
        self.seq_counter = Counter()
        self.inputs: List[Tuple[RequestGroup, SamplingParams]] = []

    def _add_query(
        self,
        token_ids: List[int],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
    ) -> None:
        if arrival_time is None:
            arrival_time = time.time()

        group_id = next(self.seq_group_counter)

        seq_id = next(self.seq_counter)
        seqs: List[Request] = [Request(seq_id, group_id, token_ids, sampling_params)]
        # for _ in range(sampling_params.n):
        #     seq_id = next(self.seq_counter)
        #     seq = Request(seq_id, group_id, token_ids, sampling_params.max_num_steps)
        #     seqs.append(seq)

        seq_group = RequestGroup(group_id, seqs, arrival_time)
        self.inputs.append((seq_group, sampling_params))

    def get_inputs(self) -> List[Tuple[RequestGroup, SamplingParams]]:
        inputs = self.inputs
        self.inputs = []
        return inputs


def get_opt_config(model_name: str):
    if "175b" in model_name:
        hf_config = vars(AutoConfig.from_pretrained(model_name.replace("175b", "30b")))
        return {"num_attention_heads": 96,
                "num_hidden_layers": 96,
                "bos_token_id": hf_config["bos_token_id"],
                "hidden_size": 12288,
                "do_layer_norm_before": hf_config["do_layer_norm_before"],
                "activation_function": hf_config["activation_function"],
                "max_position_embeddings": hf_config["max_position_embeddings"],
                "vocab_size": hf_config["vocab_size"],
               }


class FTHandler:

    def __init__(self, model_name, ft_model_location,
                 tensor_para_size, pipeline_para_size,
                 lib_path, data_type, weights_data_type):
        self.model_name = model_name
        if "175b" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name.replace("175b", "30b"))
            hf_config = get_opt_config(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            hf_config = vars(AutoConfig.from_pretrained(model_name))
        self.ckpt_path = os.path.join(ft_model_location, f'{tensor_para_size}-gpu')


        head_num = hf_config['num_attention_heads']
        layer_num = hf_config['num_hidden_layers']
        start_id = hf_config['bos_token_id']
        # end_id = hf_config['eos_token_id']
        end_id = -10000
        size_per_head = hf_config['hidden_size'] // head_num
        layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
        activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
        has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'

        max_seq_len = hf_config['max_position_embeddings']
        vocab_size = hf_config['vocab_size']

        self.gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                          max_seq_len, tensor_para_size, pipeline_para_size, lib_path,
                          inference_data_type=data_type,
                          layernorm_eps=1e-5,
                          layernorm_type=layernorm_type,
                          activation_type=activation_type,
                          has_post_decoder_layernorm=has_post_decoder_layernorm,
                          int8_mode=0,
                          weights_data_type=weights_data_type)
        if not self.gpt.load(ckpt_path=self.ckpt_path):
            raise Exception("[ERROR] Checkpoint file not found.")
        print("weights loaded")
        # code.interact(local=locals())

    def forward(self, requests):
        input_ids = [req.token_ids for req in requests]
        input_lens = [len(x) for x in input_ids]
        input_len = max(input_lens)
        pad_token = self.tokenizer.pad_token_id
        for i in range(len(input_ids)):
            input_ids[i] = [pad_token] * (input_len - len(input_ids[i])) + input_ids[i]

        # TODO what if the params are different for different input?
        sampling_params = requests[0].sampling_params
        line_encoded = torch.tensor(input_ids, dtype=torch.int32)

        max_batch_size = len(input_ids)
        output_lens = [req.output_len for req in requests]

        infer_decode_args = dict(
            beam_width=sampling_params.n,
            top_k=1 * torch.ones(max_batch_size, dtype=torch.int32),
            top_p=sampling_params.top_p * torch.ones(max_batch_size, dtype=torch.float32),
            temperature=sampling_params.temperature * torch.ones(max_batch_size, dtype=torch.float32),
            repetition_penalty=1 * torch.ones(max_batch_size, dtype=torch.float32),
            random_seed=0 * torch.ones(max_batch_size, dtype=torch.int64)
        )
        # TODO: force it to generate "output_len" tokens
        start = time.time()
        with torch.no_grad():
            output, ft_output_len = self.gpt(
                    line_encoded, torch.IntTensor(input_lens),
                    max(output_lens),
                    return_output_length=True,
                    **infer_decode_args)
        torch.cuda.synchronize()
        print("FT latency", time.time() - start)

        print("output len", max(output_lens), len(output[0][0]))

        # DEBUG
        # for i in range(sampling_params.n):
        #     print(f"=========== output {i} ============")
        #     tokens = output[0][i].cpu().numpy()
        #     output_lines = self.tokenizer.decode(tokens)
        #     output_lines = ".".join(output_lines.split('.')[:4]) + "."
        #     print(output_lines)


class FTServer:
    def __init__(
        self,
        max_bs: int,
        ft: FTHandler,
    ):
        self.max_bs = max_bs
        self.ft = ft
        self.queue: List[RequestGroup] = []
        self.sampling_params: Dict[int, SamplingParams] = {}

    def add_sequence_groups(
        self,
        seq_groups: List[Tuple[RequestGroup, SamplingParams]]
    ):
        for seq_group, sampling_params in seq_groups:
            self.queue.append(seq_group)
            self.sampling_params[seq_group.group_id] = sampling_params

    def get_requests(self, max_bs):
        # batch only the requests that max(input) + max(output) <= 2048
        max_input = 0
        max_output = 0
        seqs = []
        seq_groups = []
        cnt = 0
        flag = False
        for seq_group in self.queue:
            for i, seq in enumerate(seq_group.seqs):
                if seq.finish: continue
                if len(seqs) == max_bs:
                    flag = True
                    break
                max_input = max(max_input, len(seq.token_ids))
                max_output = max(max_output, seq.output_len)
                # seqs.append(seq)
                # if i == 0 or len(seq_groups) == 0:
                #     seq_groups.append(seq_group)
                if (max_input + max_output <= 2040 and
                    # TODO not a sufficient comparison for sampling_params
                    (len(seqs) == 0 or seq.sampling_params.n == seqs[-1].sampling_params.n)):
                    seqs.append(seq)
                    if i == 0 or len(seq_groups) == 0:
                        seq_groups.append(seq_group)
                else:
                    flag = True
                    break
            if flag:
                break
            else:
                cnt += 1

        self.queue = self.queue[cnt:]
        print("batch size", len(seqs),  "num groups", cnt,
              "input len", max_input, "output len", max_output)

        return seqs, seq_groups

    def step(self):
        if len(self.queue) == 0: return []
        print("num groups in the queue:", len(self.queue))
        seqs, seq_groups = self.get_requests(self.max_bs)
        self.ft.forward(seqs)
        for seq in seqs:
            seq.finish = True
        return seq_groups

    def has_unfinished_requests(self):
        return len(self.queue) > 0

