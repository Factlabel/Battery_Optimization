import pandas as pd
import numpy as np
import os

# scikit-learn関連
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
from xgboost import XGBRegressor

# 入出力ファイルパス (必要に応じて調整してください)
INPUT_CSV_PATH = "/Users/tkshsgw/Desktop/Battery Optimization Project/JWA予測価格生成.csv"
OUTPUT_CSV_PATH = "/Users/tkshsgw/Desktop/Battery Optimization Project/JWA予測価格生成_補完.csv"

def main():
    # ==========
    # 0) CSV読み込み・前処理
    # ==========
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"ERROR: 入力ファイルが見つかりません: {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)

    # 必要な列があるかチェック
    required_cols = ["date", "slot", "JEPX_prediction", "JEPX_actual"]
    for c in required_cols:
        if c not in df.columns:
            print(f"ERROR: CSVに '{c}' 列がありません。列名を確認してください。")
            return

    # 日付型へ変換 & 曜日情報の作成
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weekday"] = df["date"].dt.weekday  # 月曜=0, 日曜=6

    # ==========
    # 1) 学習用データセット (train_df)
    # ==========
    train_df = df.dropna(subset=["JEPX_prediction"]).copy()

    if len(train_df) == 0:
        print("ERROR: 学習用データ(JEPX_predictionが入力済みの行)がありません。")
        return

    # 特徴量(X)と目的変数(y)
    # 例: JEPX_actual, slot, weekday を使用
    X_train = train_df[["JEPX_actual", "slot", "weekday"]]
    y_train = train_df["JEPX_prediction"]

    # ==========
    # 2) 各モデルの学習 & 学習データでの予測精度を計算
    # ==========

    # -- (1) 線形回帰 --
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_train)
    lr_mae = mean_absolute_error(y_train, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_train, y_pred_lr))
    lr_r2 = r2_score(y_train, y_pred_lr)

    # -- (2) ランダムフォレスト --
    #    n_jobs=-1: M1 Mac でも全CPUコアを使用
    model_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_train)
    rf_mae = mean_absolute_error(y_train, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_train, y_pred_rf))
    rf_r2 = r2_score(y_train, y_pred_rf)

    # -- (3) XGBoost --
    #    n_jobs=-1: 全CPUコアを使用 (XGBoostでは `nthread` と表記される場合もあり)
    model_xgb = XGBRegressor(n_jobs=-1, random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_train)
    xgb_mae = mean_absolute_error(y_train, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_train, y_pred_xgb))
    xgb_r2 = r2_score(y_train, y_pred_xgb)

    # ==========
    # 3) 予測精度の比較・表示
    # ==========

    print("【学習データにおける予測精度】")

    print("-------- LinearRegression --------")
    print(f"  MAE : {lr_mae:.3f}")
    print(f"  RMSE: {lr_rmse:.3f}")
    print(f"  R^2 : {lr_r2:.3f}\n")

    print("-------- RandomForestRegressor --------")
    print(f"  MAE : {rf_mae:.3f}")
    print(f"  RMSE: {rf_rmse:.3f}")
    print(f"  R^2 : {rf_r2:.3f}\n")

    print("-------- XGBRegressor --------")
    print(f"  MAE : {xgb_mae:.3f}")
    print(f"  RMSE: {xgb_rmse:.3f}")
    print(f"  R^2 : {xgb_r2:.3f}\n")

    # ==========
    # 4) 最も精度の良いモデルを選択
    #    (ここでは R^2 の大きさで判定)
    # ==========
    best_model_name = None
    best_model = None
    best_r2 = -999
    best_mae, best_rmse = None, None

    # 線形回帰を候補に
    if lr_r2 > best_r2:
        best_model_name = "LinearRegression"
        best_model = model_lr
        best_r2 = lr_r2
        best_mae = lr_mae
        best_rmse = lr_rmse

    # ランダムフォレストを候補に
    if rf_r2 > best_r2:
        best_model_name = "RandomForestRegressor"
        best_model = model_rf
        best_r2 = rf_r2
        best_mae = rf_mae
        best_rmse = rf_rmse

    # XGBoostを候補に
    if xgb_r2 > best_r2:
        best_model_name = "XGBRegressor"
        best_model = model_xgb
        best_r2 = xgb_r2
        best_mae = xgb_mae
        best_rmse = xgb_rmse

    print(f"最も精度が高いモデル: {best_model_name}")
    print(f"  MAE : {best_mae:.3f}")
    print(f"  RMSE: {best_rmse:.3f}")
    print(f"  R^2 : {best_r2:.3f}\n")

    # ==========
    # 5) 欠損している行を補完
    # ==========
    test_df = df[df["JEPX_prediction"].isna()].copy()
    if len(test_df) == 0:
        print("欠損している JEPX_prediction がありません。出力だけ行います。")
    else:
        X_test = test_df[["JEPX_actual", "slot", "weekday"]]
        y_pred_test = best_model.predict(X_test)

        # 欠損値を補完
        test_df["JEPX_prediction"] = y_pred_test
        df.update(test_df["JEPX_prediction"])

    # ==========
    # 6) CSV出力
    # ==========
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"欠損補完が完了しました。'{best_model_name}' で予測値を埋めました。")
    print(f"結果を '{OUTPUT_CSV_PATH}' に出力しました。")

if __name__ == "__main__":
    main()