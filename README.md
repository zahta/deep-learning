
---

 :bulb: *The best way to predict the future is to create it. `Alan Kay`*    

---

### :paperclip:  Beginner's Guides
  - Some beginner's guides can be found in my [Machine learning Repository](https://github.com/zata213/path2ml).

### :pencil2: PyTorch
  - **[PyTorch Installation](https://pytorch.org/)**
    -  **Python and CPU version of PyTorch on windows, using conda:**
       - Create a new environment by `conda create --name pytorch`.
       - Activate the new environment by `conda activate pytorch`.
       - Install Python by `conda install python -y`
       - Install Jupyter lab by `conda install jupyterlab -y`
       - Install some required packages, e.g., numpy by `conda install numpy -y`.
       - Run the install command `conda install pytorch torchvision cpuonly -c pytorch`.
       
    - **Verify The PyTorch Installation from Jupyter lab:**
       - To use PyTorch, type `import torch`: If you encounter with the error `[WinError 126] The specified module could not be found`, the [installation of Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) will likely solve the problem.
       - To check the version, type `torch.__version__`.
      
### :books: Books
  - [Deep Learning](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
