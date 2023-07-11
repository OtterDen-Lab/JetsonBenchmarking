install tensorflow
-- I decided to install both gpu and cpu versions to allow for more flexiblity. Future tests might not require the cpu version. I am seperating these instructions for the two versions.

--This is an ongoing project


--GPU instructions--
```
conda install -c conda-forge tensorflow-gpu
```

```
conda create -n tf tensorflow
conda activate tf
conda deactivate
```
