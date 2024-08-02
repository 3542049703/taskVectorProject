import torch


class TaskVector():
    # 传入两个检查点文件的路径
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        if vector is not None:
            self.vector = {}
            for key in vector.keys():
                if vector[key].dtype in [torch.int64, torch.uint8]:
                    continue
                self.vector[key] = vector[key]
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                # 根据路径读取参数
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()

                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:  # 跳过整数类型参数（一般表示一些固定信息，而非计算所用参数）
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]  # 任务向量为微调参数-预训练参数

    def __add__(self, other):
        # 两任务向量相加
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                if self.vector[key].dtype in [torch.int64, torch.uint8]:  # 跳过整数类型参数（一般表示一些固定信息，而非计算所用参数）
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __neg__(self):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if self.vector[key].dtype in [torch.int64, torch.uint8]:
                    continue
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} is present in the pretrained state dict but not in the task vector.")
                if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]

        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
