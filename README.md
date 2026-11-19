After downloading, activate your environment
```powershell
conda activate nnunetv2_attention
```
Then install the following libraries from the local folders.
```powershell
cd nnUNet
pip install -e .
```
```powershell
cd ..
cd dynamic-network-architectures
pip install -e .
```
```powershell
cd..
cd batchgeneratorsv2
pip install -e .
```
