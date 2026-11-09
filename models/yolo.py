import torch
import torch.nn as nn 


class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem: 640 -> 320 -> 160
        self.stem = nn.Sequential(
            ConvBN(3, 32, k=3, s=1, p=1),
            ConvBN(32, 64, k=3, s=2, p=1),  # 320x320
            ConvBN(64, 64, k=3, s=2, p=1),  # 160x160
        )
        # Stage 3 -> P3 (stride 8: 80x80)
        self.stage3 = nn.Sequential(
            ConvBN(64, 128, k=3, s=2, p=1),
            ConvBN(128, 256, k=3, s=1, p=1),  # Stub for E-ELAN 1
        )
        # Stage 4 -> P4 (stride 16: 40x40)
        self.stage4 = nn.Sequential(
            ConvBN(256, 256, k=3, s=2, p=1),
            ConvBN(256, 512, k=3, s=1, p=1),  # Stub for E-ELAN 2
        )
        # Stage 5 -> P5 (stride 32: 20x20)
        self.stage5 = nn.Sequential(
            ConvBN(512, 512, k=3, s=2, p=1),
            ConvBN(512, 1024, k=3, s=1, p=1),  # Stub for E-ELAN 3
        )

    def forward(self, x):
        x = self.stem(x)
        p3 = self.stage3(x)  # [B, 256, 80, 80]
        p4 = self.stage4(p3)  # [B, 512, 40, 40]
        p5 = self.stage5(p4)  # [B, 1024, 20, 20]
        return p3, p4, p5

class Head(nn.Module):
    def __init__(self, in_channels_list, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_outputs = num_anchors * (5 + num_classes)
        self.heads = nn.ModuleList(
            [nn.Conv2d(ch, self.num_outputs, kernel_size=1) for ch in in_channels_list]
        )

    def forward(self, feats):
        # Returns list of raw logits for each scale
        return [head(f) for head, f in zip(self.heads, feats)]

class yolov7(nn.Module):
    """the full yolov7 model"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head([256, 512, 1024], num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        outputs = self.head(feats)
        return outputs

if __name__=="__main__":
    dummy_input= torch.randn(2,3,640,640)
    model=yolov7(num_classes=8)
    outputs=model(dummy_input)

    for i,out in enumerate(outputs,1):
        print(f" Head scale {i} shape: {out.shape}")


