from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.contrast.slots import ScouterAttention, vis
from model.contrast.position_encode import build_position_encoding


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)

    if args.dataset == "MNIST":
            bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
    bone.global_pool = Identical()
    bone.fc = Identical()
    # fix_parameter(bone, [""], mode="fix")
    # fix_parameter(bone, ["layer4", "layer3"], mode="open")
    return bone


class MainModel(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features_layer1 = 64
            self.num_features_layer2 = 128
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        hidden_dim = 128
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis
        # フック用の辞書
        self.intermediate_outputs = {}

        # 各layerにフックを登録
        self.back_bone.layer1.register_forward_hook(self.hook_fn)
        self.back_bone.layer2.register_forward_hook(self.hook_fn)
        #self.back_bone.layer3.register_forward_hook(self.hook_fn)
        
        if not self.pre_train:
            self.conv1x1_layer1 = nn.Conv2d(self.num_features_layer1, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.conv1x1_layer2 = nn.Conv2d(self.num_features_layer2, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.norm = nn.BatchNorm2d(hidden_dim)
            self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
            self.slots = ScouterAttention(args, hidden_dim, num_concepts, vis=self.vis)
            self.scale = 1
            self.cls = torch.nn.Linear(num_concepts, num_classes)
        else:
            self.fc = nn.Linear(self.num_features, num_classes)
            self.drop_rate = 0
    # フック関数
    def hook_fn(self, module, input, output):
        self.intermediate_outputs[module] = output
        # print(f"Layer: {module}")
            
    def forward(self, x, weight=None, things=None):
        x = self.back_bone(x)
        features = x
        # x = x.view(x.size(0), self.num_features, self.feature_size, self.feature_size)
        
        # 各layerの出力
        layer1_output = self.intermediate_outputs[self.back_bone.layer1]
        layer2_output = self.intermediate_outputs[self.back_bone.layer2]
        #layer3_output = self.intermediate_outputs[self.back_bone.layer3]
        #print("Layer1 output:", layer1_output.shape)
        #print("Layer2 output:", layer2_output.shape)
        #print("Layer3 output:", layer3_output.shape)
        
        if not self.pre_train: # BotCLのtrainの時
            layer1_output = F.adaptive_avg_pool2d(layer1_output, (7, 7))
            layer2_output = F.adaptive_avg_pool2d(layer2_output, (7, 7))

            
           
            layer1_output = self.conv1x1_layer1(layer1_output)
            layer1_output = self.norm(layer1_output)
            layer1_output = torch.relu(layer1_output)
            l1_pe = self.position_emb(layer1_output)
            layer1_pe = layer1_output + l1_pe

            layer2_output = self.conv1x1_layer2(layer2_output)
            layer2_output = self.norm(layer2_output)
            layer2_output = torch.relu(layer2_output)
            l2_pe = self.position_emb(layer2_output)
            layer2_pe = layer2_output + l2_pe

            x = self.conv1x1(x)
            x = self.norm(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            #print("layer1_output shape:", layer1_output.shape)
            #print("layer2_output shape:", layer2_output.shape)
            #print("x shape:", x.shape)
            # layer1, layer2, x をくっつける
            new_x = torch.cat((layer1_output, layer2_output, x), dim=3)
            new_x_pe = torch.cat((layer1_pe, layer2_pe, x_pe), dim=3)
            print("new_x shape:", new_x.shape)
            print("new_x_pe shape:", new_x_pe.shape)
            b, n, r, c = new_x.shape
            new_x = new_x.reshape((b, n, -1)).permute((0, 2, 1))
            new_x_pe = new_x_pe.reshape((b, n, -1)).permute((0, 2, 1))

            updates, attn = self.slots(new_x_pe, new_x, weight, things)
            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            cpt = self.activation(attn_cls)
            cls = self.cls(cpt)
            return (cpt - 0.5) * 2, cls, attn, updates
        else:   # pre-trainの時
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            if self.drop_rate > 0:  
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            return x, features 

class MainModel2(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel2, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.back_bone = load_backbone(args)

    def forward(self, x):
        x = self.back_bone(x)
        features = x

        return features


# if __name__ == '__main__':
#     model = MainModel()
#     inp = torch.rand((2, 1, 224, 224))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)


