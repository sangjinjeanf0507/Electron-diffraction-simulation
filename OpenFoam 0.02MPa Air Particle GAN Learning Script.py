import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# 1) 안전한 데이터 로딩 함수
# -------------------------

def safe_read_csv(csv_path, sample_size=None, chunk_size=1000):
    """
    CSV 파일을 안전하게 청크 단위로 읽습니다.
    """
    try:
        # 먼저 작은 샘플로 파일 구조 확인
        test_df = pd.read_csv(csv_path, nrows=10)
        print(f"파일 구조 확인 완료: {test_df.shape}")
        
        # 매우 작은 청크로 읽기
        chunks = []
        total_rows = 0
        
        print("청크 단위로 데이터 읽는 중...")
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, on_bad_lines='skip')):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if i % 10 == 0:  # 진행 상황 출력
                print(f"읽은 청크: {i+1}, 총 행 수: {total_rows}")
            
            if sample_size and total_rows >= sample_size:
                print(f"목표 샘플 크기에 도달: {sample_size}")
                break
        
        # 청크들을 합치기
        print("청크들을 합치는 중...")
        df = pd.concat(chunks, ignore_index=True)
        
        # 샘플 크기 제한
        if sample_size and len(df) > sample_size:
            df = df.head(sample_size)
        
        print(f"데이터 로딩 완료: {df.shape}")
        return df
        
    except Exception as e:
        print(f"데이터 로딩 실패: {e}")
        print("더 작은 청크 크기로 재시도합니다...")
        
        try:
            # 매우 작은 청크 크기로 재시도
            chunks = []
            total_rows = 0
            
            for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=500, on_bad_lines='skip')):
                chunks.append(chunk)
                total_rows += len(chunk)
                
                if i % 20 == 0:  # 진행 상황 출력
                    print(f"재시도 - 읽은 청크: {i+1}, 총 행 수: {total_rows}")
                
                if sample_size and total_rows >= sample_size:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            
            if sample_size and len(df) > sample_size:
                df = df.head(sample_size)
            
            print(f"재시도 후 데이터 로딩 완료: {df.shape}")
            return df
            
        except Exception as e2:
            print(f"두 번째 시도도 실패: {e2}")
            print("가장 작은 샘플로 시도합니다...")
            
            try:
                # 가장 작은 샘플로 시도
                df = pd.read_csv(csv_path, nrows=1000, on_bad_lines='skip')
                print(f"최소 샘플 로딩 완료: {df.shape}")
                return df
            except Exception as e3:
                print(f"모든 시도 실패: {e3}")
                return None

def calculate_velocity_and_acceleration(df):
    """
    위치 데이터로부터 속도와 가속도를 계산합니다.
    """
    # 시간별로 그룹화
    df_sorted = df.sort_values(['TimeStep', 'Time'])
    
    # 속도 계산 (위치의 시간 미분)
    df_sorted['vx'] = df_sorted.groupby('TimeStep')['Points:0'].diff() / df_sorted.groupby('TimeStep')['Time'].diff()
    df_sorted['vy'] = df_sorted.groupby('TimeStep')['Points:1'].diff() / df_sorted.groupby('TimeStep')['Time'].diff()
    df_sorted['vz'] = df_sorted.groupby('TimeStep')['Points:2'].diff() / df_sorted.groupby('TimeStep')['Time'].diff()
    
    # 가속도 계산 (속도의 시간 미분)
    df_sorted['ax'] = df_sorted.groupby('TimeStep')['vx'].diff() / df_sorted.groupby('TimeStep')['Time'].diff()
    df_sorted['ay'] = df_sorted.groupby('TimeStep')['vy'].diff() / df_sorted.groupby('TimeStep')['Time'].diff()
    df_sorted['az'] = df_sorted.groupby('TimeStep')['vz'].diff() / df_sorted.groupby('TimeStep')['Time'].diff()
    
    # NaN 값들을 0으로 채움
    velocity_cols = ['vx', 'vy', 'vz', 'ax', 'ay', 'az']
    df_sorted[velocity_cols] = df_sorted[velocity_cols].fillna(0)
    
    return df_sorted

def clean_data(data):
    """
    무한대 값과 너무 큰 값을 정제합니다.
    """
    # 무한대 값을 NaN으로 변경
    data = np.where(np.isinf(data), np.nan, data)
    
    # 너무 큰 값 처리 (평균의 10배 이상인 값)
    for col in range(data.shape[1]):
        col_data = data[:, col]
        mean_val = np.nanmean(col_data)
        std_val = np.nanstd(col_data)
        
        if not np.isnan(mean_val) and not np.isnan(std_val):
            # 평균 ± 10*표준편차 범위를 벗어나는 값들을 평균으로 대체
            upper_bound = mean_val + 10 * std_val
            lower_bound = mean_val - 10 * std_val
            
            data[:, col] = np.where(
                (col_data > upper_bound) | (col_data < lower_bound),
                mean_val,
                col_data
            )
    
    # NaN 값을 평균으로 대체
    for col in range(data.shape[1]):
        col_data = data[:, col]
        mean_val = np.nanmean(col_data)
        if not np.isnan(mean_val):
            data[:, col] = np.where(np.isnan(col_data), mean_val, col_data)
    
    return data

def prepare_training_data(csv_path, sample_size=None):
    """
    학습 데이터를 준비하고 훈련/검증/테스트로 분할합니다.
    """
    print("데이터 로딩 중...")
    
    # 안전한 데이터 로딩
    df = safe_read_csv(csv_path, sample_size)
    
    if df is None:
        print("데이터 로딩 실패!")
        return None, None, None, None, None
    
    print(f"원본 데이터 크기: {df.shape}")
    
    # 속도와 가속도 계산
    print("속도와 가속도 계산 중...")
    df_processed = calculate_velocity_and_acceleration(df)
    
    # 학습에 사용할 피처 선택 (시간 변수 제외)
    feature_cols = [
        'Points:0', 'Points:1', 'Points:2',  # 위치 (x, y, z)
        'T',                                  # 온도
        'rho',                                # 밀도
        'U:0', 'U:1', 'U:2',                # 기존 속도
        'vx', 'vy', 'vz',                    # 계산된 속도
        'ax', 'ay', 'az'                     # 계산된 가속도
    ]
    
    # 필요한 컬럼이 모두 있는지 확인
    missing_cols = [col for col in feature_cols if col not in df_processed.columns]
    if missing_cols:
        print(f"누락된 컬럼: {missing_cols}")
        return None, None, None, None, None
    
    # NaN 값 처리
    df_processed = df_processed.dropna(subset=feature_cols)
    
    # 피처 데이터 추출
    X = df_processed[feature_cols].values
    
    print(f"최종 학습 데이터 크기: {X.shape}")
    print(f"사용된 피처: {feature_cols}")
    
    # 데이터 정제
    print("데이터 정제 중...")
    X = clean_data(X)
    
    # 슬라이딩 윈도우 적용
    print("슬라이딩 윈도우 적용 중...")
    df_processed_for_windows = df_processed[feature_cols].copy()
    windows = create_sliding_windows(df_processed_for_windows, window_size=1000, overlap=0.67)
    
    all_data = []
    for i, window in enumerate(windows):
        window_data = window.values
        all_data.append(window_data)
        if i % 5 == 0:  # 진행 상황 출력
            print(f"처리된 윈도우: {i+1}/{len(windows)}")
    
    X = np.vstack(all_data)
    print(f"슬라이딩 윈도우 적용 후 데이터 크기: {X.shape}")
    
    # 데이터 분할: 훈련(70%) / 검증(15%) / 테스트(15%)
    X_temp, X_test = train_test_split(X, test_size=0.15, random_state=42)
    X_train, X_val = train_test_split(X_temp, test_size=0.176, random_state=42)  # 0.176 * 0.85 ≈ 0.15
    
    print(f"훈련 데이터 크기: {X_train.shape}")
    print(f"검증 데이터 크기: {X_val.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    
    return X_train, X_val, X_test, feature_cols

def create_sliding_windows(df, window_size=1000, overlap=0.67):
    """
    슬라이딩 윈도우로 데이터를 분할합니다. (3번 겹치도록 설정)
    """
    windows = []
    step_size = int(window_size * (1 - overlap))  # 0.67 겹침 = 0.33 스텝
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window_data = df.iloc[i:i+window_size]
        windows.append(window_data)
    
    return windows

# -------------------------
# 2) 데이터셋 클래스
# -------------------------

class ParticleDataset(Dataset):
    def __init__(self, data, scaler=None, fit_scaler=True):
        # 데이터 검증
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("경고: 데이터에 NaN 또는 무한대 값이 있습니다. 정제합니다...")
            data = clean_data(data)
        
        self.data = torch.tensor(data, dtype=torch.float32)
        self.scaler = scaler
        
        # 데이터 정규화
        if fit_scaler:
            self.scaler = StandardScaler()
            # 정규화 전에 데이터 검증
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                raise ValueError("정규화 전에 데이터에 NaN 또는 무한대 값이 있습니다.")
            self.data = torch.tensor(self.scaler.fit_transform(data), dtype=torch.float32)
        else:
            self.data = torch.tensor(self.scaler.transform(data), dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------
# 3) GAN 모델 구성
# -------------------------

# DATA_DIM은 실제 데이터에 따라 동적으로 설정됨

class Generator(nn.Module):
    def __init__(self, data_dim, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, data_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# 4) 평가 함수들
# -------------------------

def calculate_metrics(real_data, generated_data, feature_names):
    """
    실제 데이터와 생성된 데이터 간의 메트릭을 계산합니다.
    """
    # 무한대 값과 큰 값 처리
    real_data = np.where(np.isinf(real_data), np.nan, real_data)
    generated_data = np.where(np.isinf(generated_data), np.nan, generated_data)
    
    # NaN 값을 평균으로 대체
    for col in range(real_data.shape[1]):
        real_mean = np.nanmean(real_data[:, col])
        gen_mean = np.nanmean(generated_data[:, col])
        
        if not np.isnan(real_mean):
            real_data[:, col] = np.where(np.isnan(real_data[:, col]), real_mean, real_data[:, col])
        if not np.isnan(gen_mean):
            generated_data[:, col] = np.where(np.isnan(generated_data[:, col]), gen_mean, generated_data[:, col])
    
    # 전체 데이터 메트릭
    mse = mean_squared_error(real_data, generated_data)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_data, generated_data)
    
    # 피처별 메트릭
    feature_metrics = {}
    for i, feature in enumerate(feature_names):
        feature_mse = mean_squared_error(real_data[:, i], generated_data[:, i])
        feature_rmse = np.sqrt(feature_mse)
        feature_mae = mean_absolute_error(real_data[:, i], generated_data[:, i])
        
        feature_metrics[feature] = {
            'MSE': feature_mse,
            'RMSE': feature_rmse,
            'MAE': feature_mae
        }
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'feature_metrics': feature_metrics
    }

def evaluate_model(generator, scaler, test_data, feature_names, device='cpu'):
    """
    모델을 테스트 데이터로 평가합니다.
    """
    print("모델 평가 중...")
    
    # 테스트 데이터 정제
    test_data = clean_data(test_data)
    
    # 생성기로 데이터 생성
    generator.eval()
    with torch.no_grad():
        # 테스트 데이터 크기만큼 노이즈 생성
        noise = torch.randn(len(test_data), generator.latent_dim).to(device)
        generated_data = generator(noise).cpu().numpy()
    
    # 생성된 데이터 정제
    generated_data = clean_data(generated_data)
    
    # 스케일러로 역정규화
    generated_data = scaler.inverse_transform(generated_data)
    
    # 테스트 데이터도 역정규화 (일관성을 위해)
    test_data_original = scaler.inverse_transform(test_data)
    
    # 메트릭 계산
    metrics = calculate_metrics(test_data_original, generated_data, feature_names)
    
    return metrics

# -------------------------
# 5) 학습 함수
# -------------------------

def train_gan(train_data, val_data, epochs=100, batch_size=128, lr=0.0002, device='cpu', 
              early_stopping_patience=10, min_epochs=20):
    """
    검증 데이터를 포함한 GAN 학습 함수
    """
    # 훈련 데이터셋
    train_dataset = ParticleDataset(train_data, fit_scaler=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 검증 데이터셋 (훈련 데이터의 스케일러 사용)
    val_dataset = ParticleDataset(val_data, scaler=train_dataset.scaler, fit_scaler=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 데이터 차원 확인 및 설정
    data_dim = train_data.shape[1]
    # 모델 초기화
    generator = Generator(data_dim=data_dim, latent_dim=100).to(device)
    discriminator = Discriminator(data_dim=data_dim).to(device)

    criterion = nn.BCELoss()
    # 옵티마이저
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    print(f"학습 시작 - 최대 에포크: {epochs}, 배치 크기: {batch_size}, 디바이스: {device}")
    print(f"데이터 차원: {data_dim}, 잠재 차원: {generator.latent_dim}")

    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_d_loss': [], 'train_g_loss': [], 'val_d_loss': [], 'val_g_loss': []}

    for epoch in range(epochs):
        # 훈련 단계
        generator.train()
        discriminator.train()
        
        train_d_losses = []
        train_g_losses = []
        
        for batch_idx, real_data in enumerate(train_dataloader):
            real_data = real_data.to(device)
            batch_size_real = real_data.size(0)

            real_labels = torch.ones(batch_size_real, 1).to(device)
            fake_labels = torch.zeros(batch_size_real, 1).to(device)

            # Discriminator 학습
            d_optimizer.zero_grad()
            outputs_real = discriminator(real_data)
            loss_real = criterion(outputs_real, real_labels)

            z = torch.randn(batch_size_real, generator.latent_dim).to(device)
            fake_data = generator(z)
            outputs_fake = discriminator(fake_data.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            d_optimizer.step()

            # Generator 학습
            g_optimizer.zero_grad()
            outputs_fake_for_G = discriminator(fake_data)
            loss_G = criterion(outputs_fake_for_G, real_labels)
            loss_G.backward()
            g_optimizer.step()

            train_d_losses.append(loss_D.item())
            train_g_losses.append(loss_G.item())

        # 검증 단계
        generator.eval()
        discriminator.eval()
        
        val_d_losses = []
        val_g_losses = []
        
        with torch.no_grad():
            for real_data in val_dataloader:
                real_data = real_data.to(device)
                batch_size_real = real_data.size(0)

                real_labels = torch.ones(batch_size_real, 1).to(device)
                fake_labels = torch.zeros(batch_size_real, 1).to(device)

                # Discriminator 평가
                outputs_real = discriminator(real_data)
                loss_real = criterion(outputs_real, real_labels)

                z = torch.randn(batch_size_real, generator.latent_dim).to(device)
                fake_data = generator(z)
                outputs_fake = discriminator(fake_data)
                loss_fake = criterion(outputs_fake, fake_labels)

                loss_D = loss_real + loss_fake

                # Generator 평가
                outputs_fake_for_G = discriminator(fake_data)
                loss_G = criterion(outputs_fake_for_G, real_labels)

                val_d_losses.append(loss_D.item())
                val_g_losses.append(loss_G.item())

        # 평균 손실 계산
        avg_train_d_loss = np.mean(train_d_losses)
        avg_train_g_loss = np.mean(train_g_losses)
        avg_val_d_loss = np.mean(val_d_losses)
        avg_val_g_loss = np.mean(val_g_losses)
        
        training_history['train_d_loss'].append(avg_train_d_loss)
        training_history['train_g_loss'].append(avg_train_g_loss)
        training_history['val_d_loss'].append(avg_val_d_loss)
        training_history['val_g_loss'].append(avg_val_g_loss)
        
        # 조기 중단 체크 (검증 손실 기준)
        current_val_loss = avg_val_d_loss + avg_val_g_loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # 최고 성능 모델 저장
            torch.save(generator.state_dict(), 'particledata4_best_generator.pth')
            torch.save(discriminator.state_dict(), 'particledata4_best_discriminator.pth')
            # pickle을 사용하여 scaler 저장
            import pickle
            with open('particledata4_best_scaler.pth', 'wb') as f:
                pickle.dump(train_dataset.scaler, f)
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train - D: {avg_train_d_loss:.4f}, G: {avg_train_g_loss:.4f}")
            print(f"  Val   - D: {avg_val_d_loss:.4f}, G: {avg_val_g_loss:.4f}")
            print(f"  Best Val Loss: {best_val_loss:.4f}")
        
        # 조기 중단 조건 체크
        if epoch >= min_epochs and patience_counter >= early_stopping_patience:
            print(f"조기 중단: {early_stopping_patience} 에포크 동안 검증 손실 개선 없음")
            break

    # 최종 모델 저장
    torch.save(generator.state_dict(), 'particledata4_final_generator.pth')
    torch.save(discriminator.state_dict(), 'particledata4_final_discriminator.pth')
    # pickle을 사용하여 scaler 저장
    import pickle
    with open('particledata4_final_scaler.pth', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    torch.save(training_history, 'particledata4_training_history.pth')
    
    print(f"학습 완료! 총 {epoch+1} 에포크 학습됨")
    print(f"최종 훈련 손실 - D: {avg_train_d_loss:.4f}, G: {avg_train_g_loss:.4f}")
    print(f"최종 검증 손실 - D: {avg_val_d_loss:.4f}, G: {avg_val_g_loss:.4f}")
    
    return generator, discriminator, train_dataset.scaler, training_history

# -------------------------
# 6) 생성 함수
# -------------------------

def generate_particles(generator, scaler, num_samples=1000, device='cpu'):
    """
    학습된 생성자로 새로운 파티클 데이터를 생성합니다.
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.latent_dim).to(device)
        fake_data = generator(z)
        
        # 정규화 해제
        fake_data_np = fake_data.cpu().numpy()
        fake_data_denorm = scaler.inverse_transform(fake_data_np)
        
        return fake_data_denorm

# -------------------------
# 7) 메인 실행
# -------------------------

if __name__ == "__main__":
    # 데이터 경로
    csv_file_path = r'C:\Users\sunma\Downloads\particledata4.csv'
    
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    # 데이터 준비 및 분할
    print("데이터 준비 및 분할 중...")
    result = prepare_training_data(csv_file_path, sample_size=5000)  # 5천 개 샘플로 시작
    
    if result[0] is None:
        print("데이터 준비 실패! 프로그램을 종료합니다.")
        exit()
    
    X_train, X_val, X_test, feature_names = result
    
    # 학습 실행
    print("\n" + "="*50)
    print("GAN 학습 시작")
    print("="*50)
    
    generator, discriminator, scaler, history = train_gan(
        X_train, X_val,
        epochs=100,  # 에포크 수 증가
        batch_size=128, 
        lr=0.0002, 
        device=device,
        early_stopping_patience=15,  # 조기 중단 인내심 증가
        min_epochs=30  # 최소 에포크 수 증가
    )
    
    # 모델 평가
    print("\n" + "="*50)
    print("모델 평가 시작")
    print("="*50)
    
    test_metrics = evaluate_model(generator, scaler, X_test, feature_names, device)
    
    # 전체 결과 출력
    print(f"\n전체 데이터 평가 결과:")
    print(f"MSE: {test_metrics['MSE']:.6f}")
    print(f"RMSE: {test_metrics['RMSE']:.6f}")
    print(f"MAE: {test_metrics['MAE']:.6f}")
    
    # 피처별 결과 출력
    print(f"\n피처별 평가 결과:")
    for feature, feature_metric in test_metrics['feature_metrics'].items():
        print(f"{feature}:")
        print(f"  MSE: {feature_metric['MSE']:.6f}")
        print(f"  RMSE: {feature_metric['RMSE']:.6f}")
        print(f"  MAE: {feature_metric['MAE']:.6f}")
    
    print("\n" + "="*50)
    print("GAN 학습 및 평가 완료!")
    print("="*50)
    
    # GAN 보간 기능 테스트
    print("\n" + "="*50)
    print("GAN 보간 기능 테스트 시작")
    print("="*50)
    
    # 최고 성능 모델 로드
    try:
        generator.load_state_dict(torch.load('particledata4_best_generator.pth'))
        try:
            scaler = torch.load('particledata4_best_scaler.pth', weights_only=False)
        except:
            import pickle
            with open('particledata4_best_scaler.pth', 'rb') as f:
                scaler = pickle.load(f)
        
        # 보간 기능 테스트
        interpolation_results = test_gan_interpolation(generator, scaler, feature_names, device)
        
        print("\n" + "="*50)
        print("GAN 보간 기능 테스트 완료!")
        print("="*50)
        
        # 보간 결과 저장
        import pickle
        with open('particledata4_gan_interpolation_results.pkl', 'wb') as f:
            pickle.dump(interpolation_results, f)
        print("보간 결과가 'particledata4_gan_interpolation_results.pkl'에 저장되었습니다.")
        
    except Exception as e:
        print(f"보간 기능 테스트 중 오류 발생: {e}")
        print("학습된 모델이 없거나 로드할 수 없습니다.")

# -------------------------
# 8) GAN 기반 입자 생성 보간 함수들
# -------------------------

def interpolate_particles_gan(generator, scaler, start_state, end_state, num_intermediate=10, device='cpu'):
    """
    GAN을 사용하여 두 파티클 상태 사이를 보간합니다.
    
    Args:
        generator: 학습된 GAN 생성기
        scaler: 데이터 정규화 스케일러
        start_state: 시작 상태 (numpy array)
        end_state: 끝 상태 (numpy array)
        num_intermediate: 중간 상태 개수
        device: 사용할 디바이스
    
    Returns:
        interpolated_states: 보간된 상태들의 리스트
    """
    generator.eval()
    
    # 입력 상태들을 정규화
    start_state_norm = scaler.transform(start_state.reshape(1, -1))
    end_state_norm = scaler.transform(end_state.reshape(1, -1))
    
    # 잠재 공간에서의 시작점과 끝점 찾기 (간단한 선형 보간 사용)
    # 실제로는 더 정교한 방법이 필요할 수 있음
    interpolated_states = []
    
    for i in range(num_intermediate + 2):  # 시작점과 끝점 포함
        alpha = i / (num_intermediate + 1)
        
        # 선형 보간
        interpolated_norm = start_state_norm * (1 - alpha) + end_state_norm * alpha
        
        # GAN으로 생성된 데이터와 보간된 데이터를 혼합
        with torch.no_grad():
            noise = torch.randn(1, generator.latent_dim).to(device)
            generated_data = generator(noise).cpu().numpy()
            
            # 보간 비율에 따른 실제 보간 데이터와 생성된 데이터를 혼합
            mix_ratio = 0.7  # 실제 보간 데이터의 비율
            mixed_data = interpolated_norm * mix_ratio + generated_data * (1 - mix_ratio)
        
        # 역정규화
        interpolated_state = scaler.inverse_transform(mixed_data)[0]
        interpolated_states.append(interpolated_state)
    
    return interpolated_states

def generate_particle_trajectory_gan(generator, scaler, initial_state, num_steps=50, noise_factor=0.1, device='cpu'):
    """
    GAN을 사용하여 파티클 궤적을 생성합니다.
    
    Args:
        generator: 학습된 GAN 생성기
        scaler: 데이터 정규화 스케일러
        initial_state: 초기 상태
        num_steps: 생성할 스텝 수
        noise_factor: 노이즈 강도
        device: 사용할 디바이스
    
    Returns:
        trajectory: 생성된 궤적
    """
    generator.eval()
    trajectory = [initial_state]
    current_state = initial_state.copy()
    
    for step in range(num_steps):
        # 현재 상태를 정규화
        current_state_norm = scaler.transform(current_state.reshape(1, -1))
        
        # GAN으로 다음 상태 생성
        with torch.no_grad():
            noise = torch.randn(1, generator.latent_dim).to(device)
            generated_data = generator(noise).cpu().numpy()
            
            # 현재 상태와 생성된 데이터를 혼합
            next_state_norm = current_state_norm * 0.8 + generated_data * 0.2
            
            # 약간의 노이즈 추가로 다양성 확보
            noise_addition = np.random.normal(0, noise_factor, next_state_norm.shape)
            next_state_norm += noise_addition
        
        # 역정규화
        next_state = scaler.inverse_transform(next_state_norm)[0]
        trajectory.append(next_state)
        current_state = next_state
    
    return np.array(trajectory)

def create_particle_animation_gan(generator, scaler, feature_names, num_particles=100, num_frames=100, device='cpu'):
    """
    GAN을 사용하여 파티클 애니메이션을 생성합니다.
    
    Args:
        generator: 학습된 GAN 생성기
        scaler: 데이터 정규화 스케일러
        feature_names: 특징 이름들
        num_particles: 생성할 파티클 수
        num_frames: 애니메이션 프레임 수
        device: 사용할 디바이스
    
    Returns:
        animation_data: 애니메이션 데이터 (frames, particles, features)
    """
    generator.eval()
    animation_data = []
    
    # 초기 파티클 상태들 생성
    initial_states = []
    for _ in range(num_particles):
        with torch.no_grad():
            noise = torch.randn(1, generator.latent_dim).to(device)
            generated_data = generator(noise).cpu().numpy()
            initial_state = scaler.inverse_transform(generated_data)[0]
            initial_states.append(initial_state)
    
    initial_states = np.array(initial_states)
    current_states = initial_states.copy()
    
    # 프레임별로 상태 업데이트
    for frame in range(num_frames):
        frame_data = []
        
        for particle_idx in range(num_particles):
            # 각 파티클의 현재 상태를 정규화
            current_state_norm = scaler.transform(current_states[particle_idx].reshape(1, -1))
            
            # GAN으로 다음 상태 생성
            with torch.no_grad():
                noise = torch.randn(1, generator.latent_dim).to(device)
                generated_data = generator(noise).cpu().numpy()
                
                # 현재 상태와 생성된 데이터를 혼합 (물리적 연속성 유지)
                next_state_norm = current_state_norm * 0.9 + generated_data * 0.1
                
                # 위치 관련 특징들에 대해서는 더 부드러운 전환
                position_indices = [i for i, name in enumerate(feature_names) if 'Points:' in name or 'x' in name or 'y' in name or 'z' in name]
                velocity_indices = [i for i, name in enumerate(feature_names) if 'vx' in name or 'vy' in name or 'vz' in name or 'U:' in name]
                
                # 속도에 따른 위치 업데이트 (간단한 물리 시뮬레이션)
                if len(velocity_indices) > 0 and len(position_indices) > 0:
                    dt = 0.01  # 시간 스텝
                    for pos_idx, vel_idx in zip(position_indices, velocity_indices):
                        if pos_idx < len(next_state_norm[0]) and vel_idx < len(next_state_norm[0]):
                            next_state_norm[0, pos_idx] += next_state_norm[0, vel_idx] * dt
            
            # 역정규화
            next_state = scaler.inverse_transform(next_state_norm)[0]
            frame_data.append(next_state)
            current_states[particle_idx] = next_state
        
        animation_data.append(np.array(frame_data))
    
    return np.array(animation_data)

def visualize_particle_interpolation(interpolated_states, feature_names, save_path=None):
    """
    보간된 파티클 상태들을 시각화합니다.
    
    Args:
        interpolated_states: 보간된 상태들
        feature_names: 특징 이름들
        save_path: 저장할 파일 경로 (선택사항)
    """
    import matplotlib.pyplot as plt
    
    # 위치 관련 특징들 찾기
    position_features = [i for i, name in enumerate(feature_names) if 'Points:' in name]
    
    if len(position_features) >= 2:  # 최소 2D 위치가 있는 경우
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D 궤적 플롯
        x_positions = [state[position_features[0]] for state in interpolated_states]
        y_positions = [state[position_features[1]] for state in interpolated_states]
        
        axes[0].plot(x_positions, y_positions, 'b-o', linewidth=2, markersize=6)
        axes[0].scatter(x_positions[0], y_positions[0], color='green', s=100, label='Start')
        axes[0].scatter(x_positions[-1], y_positions[-1], color='red', s=100, label='End')
        axes[0].set_xlabel(feature_names[position_features[0]])
        axes[0].set_ylabel(feature_names[position_features[1]])
        axes[0].set_title('Particle Trajectory (GAN Interpolation)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 속도 관련 특징들 찾기
        velocity_features = [i for i, name in enumerate(feature_names) if 'vx' in name or 'vy' in name or 'U:' in name]
        
        if len(velocity_features) >= 2:
            vx_values = [state[velocity_features[0]] for state in interpolated_states]
            vy_values = [state[velocity_features[1]] for state in interpolated_states]
            
            axes[1].plot(vx_values, vy_values, 'r-o', linewidth=2, markersize=6)
            axes[1].scatter(vx_values[0], vy_values[0], color='green', s=100, label='Start')
            axes[1].scatter(vx_values[-1], vy_values[-1], color='red', s=100, label='End')
            axes[1].set_xlabel(feature_names[velocity_features[0]])
            axes[1].set_ylabel(feature_names[velocity_features[1]])
            axes[1].set_title('Velocity Evolution (GAN Interpolation)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # 모든 특징들의 변화를 시각화
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        if i < len(axes):
            feature_values = [state[i] for state in interpolated_states]
            axes[i].plot(feature_values, 'b-', linewidth=2)
            axes[i].set_title(f'{feature_name}')
            axes[i].set_xlabel('Interpolation Step')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        base_name = save_path.rsplit('.', 1)[0]
        plt.savefig(f'{base_name}_features.png', dpi=300, bbox_inches='tight')
    
    plt.show()

# -------------------------
# 9) 보간 기능 테스트 함수
# -------------------------

def test_gan_interpolation(generator, scaler, feature_names, device='cpu'):
    """
    GAN 보간 기능을 테스트합니다.
    """
    print("\n" + "="*50)
    print("GAN 보간 기능 테스트")
    print("="*50)
    
    # 테스트용 시작 상태와 끝 상태 생성
    with torch.no_grad():
        start_noise = torch.randn(1, generator.latent_dim).to(device)
        end_noise = torch.randn(1, generator.latent_dim).to(device)
        
        start_generated = generator(start_noise).cpu().numpy()
        end_generated = generator(end_noise).cpu().numpy()
        
        start_state = scaler.inverse_transform(start_generated)[0]
        end_state = scaler.inverse_transform(end_generated)[0]
    
    print(f"시작 상태: {start_state[:5]}...")  # 처음 5개 값만 출력
    print(f"끝 상태: {end_state[:5]}...")
    
    # 보간 실행
    interpolated_states = interpolate_particles_gan(
        generator, scaler, start_state, end_state, 
        num_intermediate=20, device=device
    )
    
    print(f"보간된 상태 개수: {len(interpolated_states)}")
    
    # 궤적 생성 테스트
    trajectory = generate_particle_trajectory_gan(
        generator, scaler, start_state, 
        num_steps=30, noise_factor=0.05, device=device
    )
    
    print(f"생성된 궤적 길이: {len(trajectory)}")
    
    # 애니메이션 생성 테스트
    animation = create_particle_animation_gan(
        generator, scaler, feature_names,
        num_particles=50, num_frames=20, device=device
    )
    
    print(f"생성된 애니메이션 크기: {animation.shape}")
    
    # 시각화
    visualize_particle_interpolation(interpolated_states, feature_names, 'gan_interpolation_test.png')
    
    return {
        'interpolated_states': interpolated_states,
        'trajectory': trajectory,
        'animation': animation
    } 