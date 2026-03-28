@echo off
call C:\Users\kmani\anaconda3\Scripts\activate.bat vae_env
cd /d C:\Users\kmani\PycharmProjects\VAE_BASED_VIDEO_COMPRESSION\MS_VQ_VAE_128
python run_pipeline.py >> outputs_msvqvae_128\pipeline_master.log 2>&1
