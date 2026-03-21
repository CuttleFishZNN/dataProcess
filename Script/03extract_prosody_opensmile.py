from pathlib import Path
import pandas as pd
import opensmile

# ========= 配置区 =========
AUDIO_DIR = Path(r"E:\Project\Python\dataProcess\Script\data\math\llm_reseg_fuzzy\audio")
OUT_CSV = Path(r"E:\Project\Python\dataProcess\Script\data\math\llm_reseg_fuzzy\prosody_features.csv")

# 选 eGeMAPSv02 + Functionals
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def main():
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"在目录里没找到 wav 文件: {AUDIO_DIR}")

    rows = []
    total = len(wav_files)

    for i, wav_path in enumerate(wav_files, start=1):
        print(f"[{i}/{total}] 处理 {wav_path.name}", flush=True)

        # 提特征，返回 DataFrame
        feat = smile.process_file(str(wav_path))

        # 取第一行并转成 dict
        row = feat.iloc[0].to_dict()
        row["seg_id"] = wav_path.stem
        row["audio_path"] = str(wav_path)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 把 seg_id 放到最前面
    cols = ["seg_id", "audio_path"] + [c for c in df.columns if c not in ("seg_id", "audio_path")]
    df = df[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n完成，共提取 {len(df)} 条音频特征。")
    print(f"输出文件: {OUT_CSV}")

if __name__ == "__main__":
    main()