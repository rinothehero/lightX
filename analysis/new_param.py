import torch
from real_xai_final import MLPClassifier  # 모델 클래스를 정의해 둔 모듈을 import

# 1. 모델 인스턴스 생성 (저장할 때 사용한 동일한 아키텍처로)
model = MLPClassifier(
    vocab_size=30522,    # 예: BERT vocab size
    embed_dim=128,
    hidden_dim=256,
    num_classes=4
)

# 2. 저장된 state_dict 로드
state_dict = torch.load("models/mlp_xai_agnews_manualIG_0.8.pt", map_location="cpu")
#    * 만약 DataParallel 로 학습했다면 key 앞에 'module.' 이 붙어 있을 수 있습니다.
#    * 그럴 땐: 
#       from collections import OrderedDict
#       new_sd = OrderedDict()
#       for k, v in state_dict.items():
#           name = k.replace("module.", "")  
#           new_sd[name] = v
#       state_dict = new_sd

model.load_state_dict(state_dict)

# 3. 전체 파라미터 수와, 학습가능(grad=True) 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"전체 파라미터 개수: {total_params:,}")
print(f"학습가능 파라미터 개수: {trainable_params:,}")