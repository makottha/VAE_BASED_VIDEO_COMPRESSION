@echo off
call C:\Users\kmani\anaconda3\Scripts\activate.bat vae_env
cd /d C:\Users\kmani\VAE_BASED_VIDEO_COMPRESSION\MS_VQ_VAE_256
python run_pipeline.py >> outputs_msvqvae_256\pipeline_master.log 2>&1
