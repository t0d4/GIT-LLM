import glob
import json
import os
import random
from typing import Any, Optional, Union

import datasets
import deepspeed
import fire
import numpy as np
import polars as pl
import torch
import yaml
from git_llm.git_llama import GitLlamaConfig, GitLlamaForCausalLM
from git_llm.git_mpt import GitMptConfig, GitMptForCausalLM
from git_llm.gnn_opt import GNNOPTConfig, GNNOPTForCausalLM
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Batch, HeteroData
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

GitLLMForCausalLM = Any

INSTRUCTION_CANDIDATES = [
    "Summarize the data traffic and functionality patterns observed on this computer network.",
    "Provide a high-level overview of the dominant communication pathways and resource utilization within this network.",
    "Summarize the key protocols and applications driving activity on this computer network.",
    "Offer a condensed depiction of the typical behaviors and functions observed within this network.",
    "Enumerate the types of data flowing through and the actions being performed on this network.",
    "Provide a succinct overview of the prevalent activities transpiring across this network.",
    "Offer a description of the typical network traffic and application usage patterns observed on this system.",
    "Distill the essence of the network's behavior into a succinct report, emphasizing salient characteristics and potential anomalies.",
    "Elucidate the primary network functions and protocols employed in facilitating user interactions and data exchanges.",
    "Characterize the network's operational dynamics through a concise analysis of traffic metrics, application usage, and resource allocation.",
]


# SupervisedDataset
class SupervisedDataset(Dataset):
    """Dataset for supervised learning"""

    def __init__(
        self,
        model_name: str,
        dataset_dirname: str,
        dataframe_filename: str,
        max_length: int = 128,
    ):
        super(SupervisedDataset, self).__init__()
        self.json_datafiles = glob.glob(f"{dataset_dirname}/*.json")
        self.df = pl.read_parquet(f"{dataset_dirname}/{dataframe_filename}")
        self.feature_stats = {
            "Protocol": [8.148607616374722, 4.365238647498611],
            "Flow Duration": [21505373.62084664, 38620071.0667367],
            "Total Fwd Packets": [6.678572292654602, 499.83754801864484],
            "Total Backward Packets": [6.836452385560825, 676.1951958944544],
            "Total Length of Fwd Packets": [397.20103269460304, 13613.92287646111],
            "Total Length of Bwd Packets": [10330.4519178038, 1487551.518855612],
            "Fwd Packet Length Max": [170.59802937191523, 603.3754579426771],
            "Fwd Packet Length Min": [11.535409408368011, 56.88208169395191],
            "Fwd Packet Length Mean": [45.100791518750604, 160.09544237517886],
            "Fwd Packet Length Std": [60.74123915720519, 237.64434260418184],
            "Bwd Packet Length Max": [1749.5979759427723, 2898.8772313600102],
            "Bwd Packet Length Min": [23.49850902448466, 55.81978567295867],
            "Bwd Packet Length Mean": [574.9163508949277, 886.3962067857422],
            "Bwd Packet Length Std": [727.767657043599, 1278.5064429053614],
            "Flow Bytes/s": [982709.5046042904, 22265201.41050033],
            "Flow Packets/s": [81850.18785589066, 280111.61600749195],
            "Flow IAT Mean": [1997698.0951529408, 4954170.567419055],
            "Flow IAT Std": [5407605.053068946, 10529055.59541358],
            "Flow IAT Max": [17923485.24586176, 34655306.120365895],
            "Flow IAT Min": [171095.66285505178, 3227475.5929842866],
            "Fwd IAT Total": [21233227.55427191, 38611283.30551164],
            "Fwd IAT Mean": [3891338.7099747504, 9661907.262130573],
            "Fwd IAT Std": [7035202.164282467, 14278503.139225585],
            "Fwd IAT Max": [17807330.505612075, 34727903.69941009],
            "Fwd IAT Min": [753181.9202423869, 7588949.117041795],
            "Bwd IAT Total": [9531584.795173837, 28113440.661738154],
            "Bwd IAT Mean": [1838554.7839053825, 8141051.096174603],
            "Bwd IAT Std": [2425499.311169748, 8924451.4878789],
            "Bwd IAT Max": [6466752.524780033, 22044302.39679556],
            "Bwd IAT Min": [661517.6942824769, 6894341.695763947],
            "Fwd PSH Flags": [0.02958361560050324, 0.16943569349287801],
            "Fwd URG Flags": [7.560727765411787e-05, 0.008694920207078967],
            "Fwd Header Length": [-17146.512531654247, 11710937.15323628],
            "Bwd Header Length": [-911.5224987096358, 1078164.7709341298],
            "Fwd Packets/s": [73751.47934907237, 273997.1086047458],
            "Bwd Packets/s": [8153.649795293947, 38787.76780910029],
            "Min Packet Length": [9.52273158005097, 21.2183885055424],
            "Max Packet Length": [1802.9999939514178, 2923.013008927524],
            "Packet Length Mean": [289.1598595121686, 426.31518575295445],
            "Packet Length Std": [569.6797850621195, 924.1762715672427],
            "Packet Length Variance": [1178561.8197001894, 2535157.714716144],
            "FIN Flag Count": [0.06679650956482468, 0.24966937880786813],
            "SYN Flag Count": [0.02958361560050324, 0.16943569349287801],
            "RST Flag Count": [0.00015423884641440046, 0.012418341767585377],
            "PSH Flag Count": [0.3736128584793058, 0.4837628823914413],
            "ACK Flag Count": [0.36590998903190425, 0.4816846508425962],
            "URG Flag Count": [0.05527698474144327, 0.22852022305185404],
            "CWE Flag Count": [7.560727765411787e-05, 0.008694920207078967],
            "ECE Flag Count": [0.0001552469434497887, 0.012458852207012596],
            "Down/Up Ratio": [0.6605515500500017, 0.6649709179795937],
            "Average Packet Size": [319.05480559743114, 468.2256377037668],
            "Avg Fwd Segment Size": [45.100791518750604, 160.09544237517886],
            "Avg Bwd Segment Size": [574.9163508949348, 886.3962067857494],
            "Subflow Fwd Packets": [6.678572292654602, 499.83754801864484],
            "Subflow Fwd Bytes": [397.1711305203394, 13585.59273367086],
            "Subflow Bwd Packets": [6.836452385560825, 676.1951958944544],
            "Subflow Bwd Bytes": [10329.445839986774, 1487209.441882914],
            "Init_Win_bytes_forward": [6884.511062856866, 12621.774069534258],
            "Init_Win_bytes_backward": [1138.028486806026, 6317.615566522921],
            "act_data_pkt_fwd": [3.5639062953643665, 443.755590874133],
            "min_seg_size_forward": [-1597.0139198038646, 933643.4415869396],
            "Active Mean": [91678.23376173584, 674841.6478783253],
            "Active Std": [31508.059288757584, 365853.06166950404],
            "Active Max": [137976.10776254878, 935204.571372991],
            "Active Min": [72609.30250774219, 620893.9016245442],
            "Idle Mean": [16819298.773389965, 33852847.04602872],
            "Idle Std": [962361.9206674964, 6654896.943241148],
            "Idle Max": [17551255.71322462, 34735242.44776471],
            "Idle Min": [16111871.063147198, 33652557.28816393],
        }
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")

    def __len__(self) -> int:
        return len(self.json_datafiles)

    def __getitem__(self, index) -> dict:
        # load text
        json_filename = self.json_datafiles[index]
        with open(json_filename) as jf:
            json_dict = json.load(jf)

        instruction = random.choice(INSTRUCTION_CANDIDATES)
        question = ""
        answer = json_dict["response"]
        text = f"##Instruction: {instruction} ##Question: {question} ##Answer: {answer}"

        # load graph
        indices = list(map(int, json_dict["referenced_flow_indices"]))
        graph, _ = self.construct_hetero_data(
            df=self.df[indices],
            onehot_columns=[
                "is_dest_port_21",
                "is_dest_port_22",
                "is_dest_port_80",
                "is_dest_port_443",
                "is_dest_port_444",
                "is_dest_port_445",
                "is_dest_port_139",
                "is_dest_port_8080",
            ],
            unused_features=[
                "Flow ID",  # ex) 192.168.10.5-104.16.207.165-54865-443-6
                "Source IP",
                "Destination IP",
                "Source Port",
                "Destination Port",  # Destination port is already converted to one-hot feature
                "Timestamp",  # ex) 2017-07-07 03:30:00
                "Fwd Header Length_duplicated_0",  # duplicated column
                "Bwd PSH Flags",
                "Bwd URG Flags",
                "Fwd Avg Bytes/Bulk",
                "Fwd Avg Packets/Bulk",
                "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk",
                "Bwd Avg Packets/Bulk",
                "Bwd Avg Bulk Rate",
            ],
            label_mapping={
                "BENIGN": 0,
                "Bot": 1,
                "DDoS": 2,
                "DoS": 3,
                "FTP-Patator": 4,
                "PortScan": 5,
                "SSH-Patator": 6,
                "Web": 7,
            },
            statistics=self.feature_stats,
        )

        # generate a whole dataset by combining text and graph
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        # batch size 1 -> unbatch
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        inputs["graph_data"] = graph
        return inputs

    def construct_hetero_data(
        self,
        df: pl.DataFrame,  # this dataframe should contain only features+label
        onehot_columns: list[str],
        unused_features: list[str],
        label_mapping: dict[str, int],
        statistics: Optional[dict[str, tuple[float, float]]] = None,
    ) -> tuple[HeteroData, dict[str, int]]:
        ### extract label info
        labels = []
        for label in df["NewLabel"]:
            labels.append(
                [
                    1 if label_mapping[label] == i else 0
                    for i in range(len(label_mapping))
                ]
            )
        labels = np.array(labels, dtype=np.float32)

        ### extract features
        df_features = df.drop("NewLabel", *unused_features)
        for col in df_features.columns:
            if col in onehot_columns:
                continue
            if statistics:
                mean, stddev = statistics[col]
                df_features = df_features.with_columns(
                    (pl.col(col) - mean) / (stddev + 1e-5)
                )
        features = df_features.to_numpy()

        # mapping (ipaddr to host)
        all_ip_addrs = pl.concat(items=[df["Source IP"], df["Destination IP"]])
        ip2hostid = {ip: i for i, ip in enumerate(all_ip_addrs.unique())}

        ### extract edge_index
        appeared_hosts = set()
        pointing_host2flow = []
        pointed_host2flow = []
        pointing_flow2host = []
        pointed_flow2host = []
        pointing_flow2flow = []
        pointed_flow2flow = []
        last_srcip = ""
        last_destip = ""
        last_flow_idx = ""
        last_timestamp = None
        # IMPORTANT: Actually, "Flow ID" column is not unique. So we simply use index as Flow ID
        for flow_idx, (srcip, destip, timestamp) in enumerate(
            zip(df["Source IP"], df["Destination IP"], df["Timestamp"])
        ):
            # flow => subsequent flow
            if all([srcip == last_srcip, destip == last_destip]):
                assert (
                    timestamp >= last_timestamp
                ), "Flows are not sorted according to the timestamp."
                pointing_flow2flow.append(last_flow_idx)
                pointed_flow2flow.append(flow_idx)

            # src => flow
            pointing_host2flow.append(ip2hostid[srcip])
            pointed_host2flow.append(flow_idx)
            appeared_hosts.add(ip2hostid[srcip])
            # flow => dest
            pointing_flow2host.append(flow_idx)
            pointed_flow2host.append(ip2hostid[destip])
            appeared_hosts.add(ip2hostid[destip])

            # update
            last_flow_idx = flow_idx
            last_srcip = srcip
            last_destip = destip
            last_timestamp = timestamp

        host_sends = torch.stack(
            [
                torch.from_numpy(np.array(pointing_host2flow, dtype=int)),
                torch.from_numpy(np.array(pointed_host2flow, dtype=int)),
            ],
            dim=0,
        )
        flow_reaches = torch.stack(
            [
                torch.from_numpy(np.array(pointing_flow2host, dtype=int)),
                torch.from_numpy(np.array(pointed_flow2host, dtype=int)),
            ],
            dim=0,
        )
        flow_precedes = torch.stack(
            [
                torch.from_numpy(np.array(pointing_flow2flow, dtype=int)),
                torch.from_numpy(np.array(pointed_flow2flow, dtype=int)),
            ],
            dim=0,
        )

        ### construct HeteroData
        data = HeteroData()
        # node info
        data["host"].node_id = torch.from_numpy(np.array(list(appeared_hosts)))
        data["flow"].node_id = torch.arange(len(df))  # id for flow is simply an index
        data["flow"].x = torch.from_numpy(features).to(torch.float)
        data["flow"].y = torch.from_numpy(labels).to(torch.float)
        # edge info
        data["host", "sends", "flow"].edge_index = host_sends
        data["flow", "reaches", "host"].edge_index = flow_reaches
        data["flow", "precedes", "flow"].edge_index = flow_precedes

        # TODO: evaluate effectiveness of this operation
        # data = tgT.ToUndirected()(data)

        return data, ip2hostid


class MyDataCollator:
    def __init__(self):
        pass

    def __call__(self, examples: list[dict]):
        dict_keys = examples[0].keys()
        return {
            key: (
                Batch.from_data_list([example[key] for example in examples])
                if key == "graph_data"
                else torch.vstack([example[key] for example in examples])
            )
            for key in dict_keys
        }


def load_model(
    model_name: str,
    vision_model_name: Optional[str],
    num_image_with_embedding: Optional[int],
) -> GNNOPTForCausalLM:
    """Loading a GIT-LLM depending on configs"""
    if "opt" in model_name:
        gnn_config = GNNOPTConfig.from_pretrained(model_name)
        model = GNNOPTForCausalLM.from_pretrained(model_name, config=gnn_config)
        return model
    raise Exception("Only OPT is supported")
    if "llama" in model_name:
        git_config = GitLlamaConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=vision_model_name,
        )
        model = GitLlamaForCausalLM.from_pretrained(model_name, config=git_config)
    elif "mpt" in model_name:
        git_config = GitMptConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding,
            vision_model_name=vision_model_name,
        )
        model = GitMptForCausalLM.from_pretrained(model_name, config=git_config)


def load_pretrained_weight(model: GitLLMForCausalLM, weight_path: str):
    import glob

    weight = {}
    weight_path = glob.glob(f"{weight_path}/pytorch*.bin")
    for w in weight_path:
        weight_temp = torch.load(w, map_location="cpu")
        weight.update(weight_temp)
    model.load_state_dict(weight, strict=False)


def apply_lora_model(
    model: GitLLMForCausalLM, model_name: str, config: dict
) -> GitLLMForCausalLM:
    """Apply LoRA"""
    peft_config = LoraConfig(**config["lora"])
    # apply lora only to LLM
    if "opt" in model_name:
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)
    elif "llama" in model_name:
        target_modules = []
        for m in peft_config.target_modules:
            target_modules += [
                f"model.layers.{i}.self_attn.{m}"
                for i in range(len(model.model.layers))
            ]

        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model
    elif "mpt" in model_name:
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model
    return model


def set_trainable_params(
    model: GitLLMForCausalLM, model_name: str, keys_finetune: list
) -> None:
    if "mpt" in model_name:
        for name, p in model.transformer.named_parameters():
            if np.any([k in name for k in keys_finetune]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for name, p in model.model.named_parameters():
            if np.any([k in name for k in keys_finetune]):
                print("Will finetune:", name)
                p.requires_grad = True
            else:
                p.requires_grad = False


def get_dataset(config: dict) -> Union[Dataset, Dataset]:
    if config.get("dataset_type") is not None:
        dataset_list = [
            datasets.load_dataset("MMInstruction/M3IT", i)
            for i in config["dataset_type"]
        ]
        train_dataset = ConcatDataset([d["train"] for d in dataset_list])
        # some dataset have no validation
        for d in dataset_list:
            val_dataset_list = []
            try:
                val_dataset_list.append(d["validation"])
            except:
                print(f"{d['train']._info.config_name} has no validation set.")
        val_dataset = ConcatDataset(val_dataset_list)
    else:
        coco_datasets = datasets.load_dataset("MMInstruction/M3IT", "coco")
        train_dataset = coco_datasets["train"]
        val_dataset = coco_datasets["validation"]
    return train_dataset, val_dataset


def main(config_file: str, local_rank: int):
    # get config
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)

    if os.environ["WANDB_NAME"] is not None:
        config["training"]["output_dir"] = os.path.join(
            config["training"]["output_dir"], os.environ["WANDB_NAME"]
        )

    # distributed learning
    deepspeed.init_distributed(rank=local_rank)

    # model
    model_name = config["settings"]["model_name"]
    # vision_model_name = config["settings"]["vision_model_name"]
    # num_image_with_embedding = config["settings"]["num_image_with_embedding"]

    # DatasetのLoad
    train_dataset = SupervisedDataset(
        model_name=model_name,
        dataset_dirname="/tmp/train_flows",
        dataframe_filename="train_df.parquet",
    )
    val_dataset = SupervisedDataset(
        model_name=model_name,
        dataset_dirname="/tmp/val_flows",
        dataframe_filename="val_df.parquet",
    )

    # configの割り当て
    max_length = config["settings"]["max_length"]
    keys_finetune = config["settings"]["keys_finetune"]

    # 訓練に関するconfig
    training_args = TrainingArguments(**config["training"])

    # load model
    model = load_model(
        model_name=model_name,
        vision_model_name=None,
        num_image_with_embedding=None,
    )

    # lora
    if config["use_lora"]:
        keys_finetune.append("lora")
        model = apply_lora_model(model, model_name, config)

    # load pretrained weight
    # if config["settings"]["load_pretrained"] is not None:
    #     load_pretrained_weight(model, config["settings"]["load_pretrained"])
    #     print(
    #         f'Successfully loading pretrained weights from {config["settings"]["load_pretrained"]}'
    #     )

    # Set trainable params
    set_trainable_params(model, model_name, keys_finetune)

    my_data_collator = MyDataCollator()

    trainer = Trainer(
        data_collator=my_data_collator,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    with torch.autocast("cuda"):
        result = trainer.train()

    # Save the finel checkpoint
    # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2281
    final_save_path = os.path.join(
        config["training"]["output_dir"], os.environ["WANDB_NAME"] + "_final"
    )
    trainer.save_model(final_save_path)
    if "zero3" in config["training"]["deepspeed"]:
        # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        # config `stage3_gather_16bit_weights_on_model_save` is True
        trainer.model_wrapped.save_checkpoint(final_save_path)


if __name__ == "__main__":
    fire.Fire(main)
