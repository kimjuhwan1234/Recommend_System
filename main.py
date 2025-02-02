import pickle
import warnings
from Module.deepFM import *
from Module.dataset import *
from Module.trainer import *
from utils.parser import config
from deepctr_torch.inputs import SparseFeat

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    file_path = "Database/train_val_test.pkl"
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    x_train = loaded_data["x_train"]
    y_train = loaded_data["y_train"]
    x_val = loaded_data["x_val"]
    y_val = loaded_data["y_val"]
    x_test = loaded_data["x_test"]
    y_test = loaded_data["y_test"]

    print("\nPreprocessing file: import complete.")

    # 범주형(카테고리형) 피처 리스트
    sparse_features = ['userID_x', 'articleID', 'userRegion_x', 'userCountry_x',
                       'Format', 'Language', 'userID_y', 'userCountry_y', 'userRegion_y']

    # 각 범주형 피처의 vocabulary_size를 적절히 설정 (여기선 대략 5000으로 가정)
    # embedding_dim은 8로 설정 (조절 가능)
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=5000, embedding_dim=8) for feat in sparse_features]

    # FM과 DNN에 같은 피처 사용
    linear_feature_columns = sparse_feature_columns
    dnn_feature_columns = sparse_feature_columns

    # DeepFM 모델 생성
    model = DeepFM(linear_feature_columns=linear_feature_columns,
                   dnn_feature_columns=dnn_feature_columns, device='cuda:0')

    config['model'] = model

    dataloaders = get_dataloder(config, x_train, x_val, x_test, y_train, y_val, y_test)

    trainer = Trainer(config, dataloaders)
    trainer.run_model()
    trainer.check_validation()
    trainer.evaluate_testset()
