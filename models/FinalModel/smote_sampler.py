import numpy as np
import pandas as pd
from typing import Sequence, Union
from imblearn.base import BaseSampler
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# =========================================================
# 0) 커스텀 오버샘플러: MajorityVoteSMOTENC
#    - 수치형: 선형보간
#    - 범주형: k-이웃 다수결(동률 랜덤)
#    - 반드시 전처리(OneHot) 이전에 배치!
# =========================================================
class MajorityVoteSMOTENC(BaseSampler):
    """
    수치형: 랜덤 이웃과 선형 보간
    범주형: k-이웃 다수결(동률 랜덤)
    ※ 반드시 전처리(OneHot) 이전 단계에서 사용
    """
    def __init__(self,
                 categorical_features: Sequence[Union[int, str]],
                 k_neighbors: int = 5,
                 sampling_strategy="auto",
                 random_state: int = 42):
        super().__init__(sampling_strategy=sampling_strategy)
        # ✅ 파라미터를 그대로 저장 (list() 변환 제거)
        self.categorical_features = categorical_features
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _validate_and_indexify(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for MajorityVoteSMOTENC.")
        
        # ✅ 사용 시점에 list로 변환
        cat_features = list(self.categorical_features)
        
        if isinstance(cat_features[0], str):
            cat_idx = [X.columns.get_loc(c) for c in cat_features]
        else:
            cat_idx = cat_features
        
        all_idx = np.arange(X.shape[1])
        num_idx = [i for i in all_idx if i not in cat_idx]
        return cat_idx, num_idx

    # ★ BaseSampler가 요구하는 추상 메서드 구현
    def _fit_resample(self, X, y):
        rs = check_random_state(self.random_state)

        # DataFrame/Series 보장
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        # 범주/수치 분리
        cat_idx, num_idx = self._validate_and_indexify(X)

        # 클래스 분포
        class_counts = y.value_counts()
        maj_class = class_counts.idxmax()

        # 증강 대상/수 결정
        if isinstance(self.sampling_strategy, (float, int)):
            min_class = class_counts.idxmin()
            desired_min = int(np.floor(class_counts[maj_class] * float(self.sampling_strategy)))
            n_new = max(0, desired_min - class_counts[min_class])
            target_class = min_class
        elif isinstance(self.sampling_strategy, dict):
            (target_class, n_new) = list(self.sampling_strategy.items())[0]
        else:  # "auto": 최소 클래스 → 다수와 동일하게
            min_class = class_counts.idxmin()
            n_new = class_counts[maj_class] - class_counts[min_class]
            target_class = min_class

        if n_new <= 0:
            return X, y  # 컬럼명 유지

        # 소수 클래스 서브셋
        X_min = X[y == target_class].reset_index(drop=True)

        # 이웃 탐색 임베딩(수치형 우선, 없으면 범주형 OHE)
        if len(num_idx) == 0:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_emb = ohe.fit_transform(X_min.iloc[:, cat_idx])
        else:
            X_emb = X_min.iloc[:, num_idx].to_numpy()

        nn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, len(X_min)),
            metric="euclidean"
        )
        nn.fit(X_emb)
        neigh_idx = nn.kneighbors(X_emb, return_distance=False)

        synth_rows = []
        for _ in range(n_new):
            i = rs.randint(0, len(X_min))
            neighs = neigh_idx[i][1:] if neigh_idx.shape[1] > 1 else neigh_idx[i]
            j = i if np.size(neighs) == 0 else rs.choice(neighs)

            xi = X_min.iloc[i, :].copy()
            xj = X_min.iloc[j, :].copy()
            x_new = xi.copy()

            # 1) 수치형 보간
            if len(num_idx) > 0:
                lam = rs.rand()
                x_new.iloc[num_idx] = xi.iloc[num_idx] + lam * (xj.iloc[num_idx] - xi.iloc[num_idx])

            # 2) 범주형 다수결(동률 랜덤)
            if len(cat_idx) > 0:
                neigh_cats = X_min.iloc[neighs, cat_idx] if np.size(neighs) > 0 else X_min.iloc[[i], cat_idx]
                for col_pos, col_idx_val in enumerate(cat_idx):
                    counts = neigh_cats.iloc[:, col_pos].value_counts()
                    max_count = counts.max()
                    modes = counts[counts == max_count].index.tolist()
                    chosen = rs.choice(modes)
                    x_new.iloc[col_idx_val] = chosen

            synth_rows.append(x_new)

        X_syn = pd.DataFrame(synth_rows, columns=X.columns)
        y_syn = pd.Series([target_class] * len(X_syn), name=y.name)

        X_res = pd.concat([X, X_syn], axis=0, ignore_index=True)
        y_res = pd.concat([y, y_syn], axis=0, ignore_index=True)
        return X_res, y_res

    # (선택) 구버전 호환: fit_resample를 직접 호출하는 코드가 있을 경우 대비
    def fit_resample(self, X, y):
        return self._fit_resample(X, y)