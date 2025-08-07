"""
streamlit run foma_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import copy
import umap
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# 數據品質檢查函數
def data_quality_check(X, y, X_aug=None, y_aug=None):
    quality_info = {
        "數據形狀": {
            "原始": f"X={X.shape}, y={y.shape}",
            "擴增": f"X={X_aug.shape}, y={y_aug.shape}" if X_aug is not None else None
        },
        "特徵統計": {
            "原始": {
                "平均值": np.mean(X, axis=0),
                "標準差": np.std(X, axis=0)
            },
            "擴增": {
                "平均值": np.mean(X_aug, axis=0) if X_aug is not None else None,
                "標準差": np.std(X_aug, axis=0) if X_aug is not None else None
            }
        },
        "標籤統計": {
            "原始": {
                "範圍": f"[{np.min(y):.3f}, {np.max(y):.3f}]",
                "標準差": f"{np.std(y):.3f}"
            },
            "擴增": {
                "範圍": f"[{np.min(y_aug):.3f}, {np.max(y_aug):.3f}]" if y_aug is not None else None,
                "標準差": f"{np.std(y_aug):.3f}" if y_aug is not None else None
            }
        }
    }
    
    # 檢查特徵相關性
    df = pd.DataFrame(X)
    correlation_matrix = df.corr()
    
    # 獲取非1的最大相關係數
    corr_values = correlation_matrix.abs().values
    np.fill_diagonal(corr_values, 0)  # 將對角線設為0
    max_corr = np.max(corr_values)
    max_corr_indices = np.where(corr_values == max_corr)
    max_corr_features = (df.columns[max_corr_indices[0][0]], df.columns[max_corr_indices[1][0]])
    
    quality_info["特徵相關性"] = {
        "最大相關係數": f"{max_corr:.3f}",
        "相關特徵對": f"{max_corr_features[0]} - {max_corr_features[1]}",
        "相關矩陣": correlation_matrix
    }
    
    if X_aug is not None:
        df_aug = pd.DataFrame(X_aug)
        correlation_matrix_aug = df_aug.corr()
        corr_values_aug = correlation_matrix_aug.abs().values
        np.fill_diagonal(corr_values_aug, 0)
        max_corr_aug = np.max(corr_values_aug)
        max_corr_indices_aug = np.where(corr_values_aug == max_corr_aug)
        max_corr_features_aug = (df_aug.columns[max_corr_indices_aug[0][0]], df_aug.columns[max_corr_indices_aug[1][0]])
        
        quality_info["特徵相關性"]["擴增"] = {
            "最大相關係數": f"{max_corr_aug:.3f}",
            "相關特徵對": f"{max_corr_features_aug[0]} - {max_corr_features_aug[1]}",
            "相關矩陣": correlation_matrix_aug
        }
    
    return quality_info

# 計算模型參數量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 基本FOMA實現
class SimpleFOMATransform:
    def __init__(self, alpha=2.0, k=1):
        self.alpha = alpha
        self.k = k
    
    def scale(self, A, k, lam):
        U, s, Vt = torch.linalg.svd(A, full_matrices=False)
        lam_repeat = lam.repeat(s.shape[-1] - k)
        lam_ones = torch.ones(k, device=A.device, dtype=A.dtype)
        scale_factors = torch.cat((lam_ones, lam_repeat))
        s_scaled = s * scale_factors
        A_scaled = U @ torch.diag_embed(s_scaled) @ Vt
        return A_scaled
    
    def augment_batch(self, X, Y):
        device = X.device
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().to(device)
        A = torch.cat((X, Y), dim=1)
        A_scaled = self.scale(A, self.k, lam)
        n_features = X.shape[1]
        X_aug = A_scaled[:, :n_features]
        Y_aug = A_scaled[:, n_features:]
        return X_aug, Y_aug

# 增強版FOMA實現
class EnhancedFOMATransform(SimpleFOMATransform):
    def __init__(self, alpha=2.0, k=2, mode='adaptive'):
        super().__init__(alpha, k)
        self.mode = mode
        self.current_alpha = alpha
    
    def update_alpha(self, epoch, total_epochs):
        if self.mode == 'adaptive':
            progress = epoch / total_epochs
            self.current_alpha = self.alpha * (1 - 0.5 * progress)
        elif self.mode == 'cosine':
            progress = epoch / total_epochs
            self.current_alpha = self.alpha * (0.5 + 0.5 * np.cos(np.pi * progress))
    
    def augment_batch(self, X, Y):
        device = X.device
        lam = torch.distributions.beta.Beta(self.current_alpha, self.current_alpha).sample().to(device)
        A = torch.cat((X, Y), dim=1)
        A_scaled = self.scale(A, self.k, lam)
        n_features = X.shape[1]
        X_aug = A_scaled[:, :n_features]
        Y_aug = A_scaled[:, n_features:]
        return X_aug, Y_aug

# 資料集類
class RegressionDataset(Dataset):
    def __init__(self, X, y, device='cpu'):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 模型類
class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, use_batchnorm=False, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 處理每一層的設定
        for hidden_dim, activation in hidden_dims:
            # 添加線性層
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 添加BatchNorm
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 添加激活函數
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif activation == 'GELU':
                layers.append(nn.GELU())
            elif activation == 'SiLU':
                layers.append(nn.SiLU())
            
            # 添加Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 添加輸出層
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 訓練函數
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=10):
    # 確保模型在正確的設備上
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mape': [],
        'val_mape': []
    }
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_loss = 0
        train_mape = 0
        for X_batch, y_batch in train_loader:
            # 確保數據在正確的設備上
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mape += mean_absolute_percentage_error(y_batch.cpu().numpy(), y_pred.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_mape /= len(train_loader)
        
        # 驗證階段
        model.eval()
        val_loss = 0
        val_mape = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # 確保數據在正確的設備上
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
                val_mape += mean_absolute_percentage_error(y_batch.cpu().numpy(), y_pred.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_mape /= len(val_loader)
        
        # 更新歷史記錄
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mape'].append(train_mape)
        history['val_mape'].append(val_mape)
        
        # 早停檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 恢復最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# 資料擴增函數
def augment_data(X, y, foma_transform, n_augmentations=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(device)
    
    X_augmented = []
    y_augmented = []
    
    for _ in range(n_augmentations):
        X_aug, y_aug = foma_transform.augment_batch(X_tensor, y_tensor)
        X_augmented.append(X_aug.cpu().numpy())
        y_augmented.append(y_aug.cpu().numpy())
    
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.vstack(y_augmented).flatten()
    
    return X_augmented, y_augmented

# PCA降維分析
def pca_analysis(X, X_aug, feature_names):
    # PCA分析
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_aug_pca = pca.transform(X_aug)
    
    # 創建PCA圖
    fig = go.Figure()
    
    # 添加原始數據點
    fig.add_trace(go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers',
        name='原始數據',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.5
        )
    ))
    
    # 添加擴增數據點
    fig.add_trace(go.Scatter(
        x=X_aug_pca[:, 0],
        y=X_aug_pca[:, 1],
        mode='markers',
        name='擴增數據',
        marker=dict(
            size=3,
            color='red',
            opacity=0.3
        )
    ))
    
    fig.update_layout(
        title='PCA降維分析',
        xaxis_title='主成分1',
        yaxis_title='主成分2',
        showlegend=True
    )
    
    return fig

# UMAP分析
def umap_analysis(X, X_aug, feature_names):
    # 標準化數據
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_aug_scaled = scaler.transform(X_aug)
    
    # UMAP降維
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    X_aug_umap = reducer.transform(X_aug_scaled)
    
    # 創建UMAP圖
    fig = go.Figure()
    
    # 添加原始數據點
    fig.add_trace(go.Scatter(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        mode='markers',
        name='原始數據',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.5
        )
    ))
    
    # 添加擴增數據點
    fig.add_trace(go.Scatter(
        x=X_aug_umap[:, 0],
        y=X_aug_umap[:, 1],
        mode='markers',
        name='擴增數據',
        marker=dict(
            size=3,
            color='red',
            opacity=0.3
        )
    ))
    
    fig.update_layout(
        title='UMAP降維分析',
        xaxis_title='UMAP1',
        yaxis_title='UMAP2',
        showlegend=True
    )
    
    return fig

def plot_training_history(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'MAPE'))
    
    # Loss圖
    fig.add_trace(
        go.Scatter(y=history['train_loss'], name='訓練集', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_loss'], name='驗證集', line=dict(color='orange')),
        row=1, col=1
    )
    
    # MAPE圖
    fig.add_trace(
        go.Scatter(y=history['train_mape'], name='訓練集', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history['val_mape'], name='驗證集', line=dict(color='orange')),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    return fig

# Streamlit應用
def main():
    st.title('FOMA資料擴增與模型比較')
    
    # 檔案上傳
    uploaded_file = st.file_uploader("上傳Excel檔案", type=['xlsx'])
    
    if uploaded_file is not None:
        # 讀取資料
        df = pd.read_excel(uploaded_file)
        
        # 特徵選擇
        feature_cols = st.multiselect(
            "選擇特徵欄位",
            df.columns.tolist(),
            default=df.columns[:-1].tolist()
        )
        
        target_col = st.selectbox(
            "選擇目標欄位",
            df.columns.tolist(),
            index=len(df.columns)-1
        )
        
        if feature_cols and target_col:
            # 資料預處理
            X = df[feature_cols].values
            y = df[target_col].values
            
            # 數據品質檢查
            st.subheader('數據品質檢查')
            quality_info = data_quality_check(X, y)
            
            # 顯示原始數據品質
            st.write("原始數據品質：")
            st.write(quality_info["數據形狀"]["原始"])
            st.write("特徵統計：")
            st.write(f"平均值：{quality_info['特徵統計']['原始']['平均值']}")
            st.write(f"標準差：{quality_info['特徵統計']['原始']['標準差']}")
            st.write("標籤統計：")
            st.write(f"範圍：{quality_info['標籤統計']['原始']['範圍']}")
            st.write(f"標準差：{quality_info['標籤統計']['原始']['標準差']}")
            st.write(f"特徵間最大相關係數：{quality_info['特徵相關性']['最大相關係數']}")
            st.write(f"相關特徵對：{quality_info['特徵相關性']['相關特徵對']}")
            
            # 顯示相關矩陣熱圖
            st.write("特徵相關性矩陣：")
            fig = px.imshow(quality_info['特徵相關性']['相關矩陣'],
                          labels=dict(x="特徵", y="特徵", color="相關係數"),
                          x=feature_cols,
                          y=feature_cols)
            st.plotly_chart(fig)
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # 分割資料
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # 訓練參數設定
            st.subheader('訓練參數設定')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                epochs = st.slider('訓練輪數', 100, 5000, 1000)
            with col2:
                batch_size = st.slider('批次大小', 8, 256, 32)
            with col3:
                learning_rate = st.number_input('學習率', 1e-5, 1e-2, 1e-3, format='%.6f')
            with col4:
                patience = st.slider('早停耐心值', 5, 800, 10)
            
            # 模型架構設定
            st.subheader('模型架構設定')
            n_layers = st.slider('隱藏層數量', 1, 10, 5)
            hidden_dims = []
            
            # 創建一個容器來存放所有層的設定
            layer_container = st.container()
            
            with layer_container:
                for i in range(n_layers):
                    col1, col2 = st.columns(2)
                    with col1:
                        dim = st.number_input(
                            f'第{i+1}層神經元數量',
                            min_value=8,
                            max_value=1024,
                            value=128 if i == 0 else 64 if i == 1 else 32,
                            step=8
                        )
                    with col2:
                        activation = st.selectbox(
                            f'第{i+1}層激活函數',
                            ['ReLU', 'LeakyReLU', 'GELU', 'SiLU'],
                            index=1
                        )
                    hidden_dims.append((dim, activation))
            
            # 添加BatchNorm和Dropout設定
            col1, col2 = st.columns(2)
            with col1:
                use_batchnorm = st.checkbox('使用BatchNorm', value=False)
            with col2:
                dropout_rate = st.slider('Dropout率', 0.0, 0.5, 0.0, 0.1)
            
            # 計算模型參數量
            model = RegressionNet(
                input_dim=len(feature_cols),
                hidden_dims=hidden_dims,
                use_batchnorm=use_batchnorm,
                dropout_rate=dropout_rate
            )
            n_params = count_parameters(model)
            st.write(f'模型參數量：{n_params:,}')
            
            # FOMA方式選擇
            st.subheader('FOMA方式選擇')
            use_no_foma = st.checkbox('使用 No FOMA', value=True)
            use_simple_foma = st.checkbox('使用 Simple FOMA', value=True)
            use_enhanced_foma = st.checkbox('使用 Enhanced FOMA', value=True)
            
            # FOMA參數設定
            if use_simple_foma or use_enhanced_foma:
                st.subheader('FOMA參數設定')
                col1, col2, col3 = st.columns(3)
                with col1:
                    foma_k = st.slider('k值', 1, 5, 1)
                with col2:
                    foma_alpha = st.number_input('alpha值', 0.1, 5.0, 2.0, 0.1)
                with col3:
                    n_augmentations = st.slider('擴增筆數', 50, 1000, 100, 50)
            
            # 訓練模型
            if st.button('開始訓練'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.write(f'使用設備: {device}')
                
                # 創建資料加載器
                train_dataset = RegressionDataset(X_train, y_train, device=device)
                val_dataset = RegressionDataset(X_val, y_val, device=device)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # 訓練選擇的模型
                models = {}
                if use_no_foma:
                    models['No FOMA'] = None
                if use_simple_foma:
                    models['Simple FOMA'] = SimpleFOMATransform(alpha=foma_alpha, k=foma_k)
                if use_enhanced_foma:
                    models['Enhanced FOMA'] = EnhancedFOMATransform(alpha=foma_alpha, k=foma_k, mode='adaptive')
                
                results = {}
                
                for name, foma_transform in models.items():
                    st.write(f'訓練 {name} 模型...')
                    
                    model = RegressionNet(
                        input_dim=len(feature_cols),
                        hidden_dims=hidden_dims,
                        use_batchnorm=use_batchnorm,
                        dropout_rate=dropout_rate
                    ).to(device)
                    
                    best_model, history = train_model(
                        model, train_loader, val_loader,
                        criterion=nn.MSELoss(),
                        optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                        device=device,
                        epochs=epochs,
                        patience=patience
                    )
                    
                    results[name] = {
                        'model': best_model,
                        'history': history
                    }
                    
                    # 保存最佳模型
                    best_model_path = f'model_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                    torch.save(best_model.state_dict(), best_model_path)
                    
                    # 顯示訓練結果
                    st.write(f'{name} 訓練完成')
                    st.write(f'最終驗證損失: {history["val_loss"][-1]:.4f}')
                    
                    # 計算並顯示R²分數
                    best_model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_val).to(device)
                        y_pred = best_model(X_val_tensor).cpu().numpy()
                        r2 = r2_score(y_val, y_pred)
                        st.write(f'最終驗證R²: {r2:.4f}')
                    
                    st.write(f'最終驗證MAPE: {history["val_mape"][-1]:.4f}')
                    
                    # 繪製訓練過程
                    st.plotly_chart(plot_training_history(history))
                    
                    # 提供模型下載
                    with open(best_model_path, 'rb') as f:
                        st.download_button(
                            label=f'下載{name}模型',
                            data=f,
                            file_name=best_model_path,
                            mime='application/octet-stream'
                        )
                
                # 比較不同方法的結果
                if len(results) > 1:
                    st.subheader('模型比較')
                    comparison_data = []
                    for name, result in results.items():
                        comparison_data.append({
                            '方法': name,
                            '驗證損失': result['history']['val_loss'][-1],
                            '驗證MAPE': result['history']['val_mape'][-1]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.write(comparison_df)
                    
                    # 繪製比較圖表
                    fig = go.Figure()
                    for name, result in results.items():
                        fig.add_trace(go.Scatter(
                            y=result['history']['val_loss'],
                            name=f'{name} 驗證損失',
                            mode='lines'
                        ))
                    
                    fig.update_layout(
                        title='驗證損失比較',
                        xaxis_title='Epoch',
                        yaxis_title='損失',
                        showlegend=True
                    )
                    st.plotly_chart(fig)
                
                # 資料擴增
                st.subheader('資料擴增')
                
                # 使用Enhanced FOMA進行擴增
                foma_transform = EnhancedFOMATransform(alpha=foma_alpha, k=foma_k, mode='adaptive')
                X_aug, y_aug = augment_data(X_scaled, y_scaled, foma_transform, n_augmentations=n_augmentations)
                
                # 反標準化
                X_aug_original = scaler_X.inverse_transform(X_aug)
                y_aug_original = scaler_y.inverse_transform(y_aug.reshape(-1, 1)).flatten()
                
                # 顯示擴增結果
                st.write(f'原始資料數量: {len(X)}')
                st.write(f'擴增後資料數量: {len(X_aug)}')
                
                # 擴增數據品質檢查
                st.subheader('擴增數據品質檢查')
                aug_quality_info = data_quality_check(X, y, X_aug_original, y_aug_original)
                
                # 顯示擴增數據品質
                st.write("擴增數據品質：")
                st.write(aug_quality_info["數據形狀"]["擴增"])
                st.write("特徵統計：")
                st.write(f"平均值：{aug_quality_info['特徵統計']['擴增']['平均值']}")
                st.write(f"標準差：{aug_quality_info['特徵統計']['擴增']['標準差']}")
                st.write(f"標籤統計：")
                st.write(f"範圍：{aug_quality_info['標籤統計']['擴增']['範圍']}")
                st.write(f"標準差：{aug_quality_info['標籤統計']['擴增']['標準差']}")
                st.write(f"特徵間最大相關係數：{aug_quality_info['特徵相關性']['擴增']['最大相關係數']}")
                st.write(f"相關特徵對：{aug_quality_info['特徵相關性']['擴增']['相關特徵對']}")
                
                # 顯示擴增數據相關矩陣熱圖
                st.write("擴增數據特徵相關性矩陣：")
                fig = px.imshow(aug_quality_info['特徵相關性']['擴增']['相關矩陣'],
                              labels=dict(x="特徵", y="特徵", color="相關係數"),
                              x=feature_cols,
                              y=feature_cols)
                st.plotly_chart(fig)
                
                # PCA降維分析
                st.plotly_chart(pca_analysis(X, X_aug_original, feature_cols))
                
                # UMAP分析
                st.plotly_chart(umap_analysis(X, X_aug_original, feature_cols))
                
                # 下載擴增後的資料
                aug_df = pd.DataFrame(X_aug_original, columns=feature_cols)
                aug_df[target_col] = y_aug_original
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv = aug_df.to_csv(index=False)
                
                st.download_button(
                    label="下載擴增後的資料",
                    data=csv,
                    file_name=f'augmented_data_{timestamp}.csv',
                    mime='text/csv'
                )

if __name__ == '__main__':
    main() 