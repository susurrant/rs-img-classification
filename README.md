# rs-img-classification
Semantic segmentation of remote sensing images.</br>

## Development enviroment
1. Operation system: Ubuntu 16.04.3 LTS</br>
2. CUDA version: 8.0</br>
3. Programming language: Python 2.7</br>
   Modules: </br>
   tensorflow-gpu==1.4.1, scikit-image==0.13.0, tifffile==0.13.5,</br>
   pandas==0.20.3, keras==1.2.2, h5py==2.7.0, tqdm== 4.19.4</br>
   
## Model
- Reference: [kaggle-dstl 3rd blog](http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/)</br>
- Framework:</br>
![Model structure](screenshots/model.jpg)</br>
- Input and output: 1024\*1024\*3 (in), 1000\*1000\*1 (out), row\*column\*channel

## Result
left: image, middle: ground truth(label), right: prediction</br>
1. Airport
![airport238](screenshots/airport238.JPG)</br>
![airport238](screenshots/airport352.JPG)</br>
2. Baresoil
![baresoil542](screenshots/baresoil542.JPG)</br>
![baresoil7181](screenshots/baresoil7181.JPG)</br>
3. Building
![building1675](screenshots/build1675.JPG)</br>
![building6634](screenshots/build6634.JPG)</br>
4. Farmland
![farmland2833](screenshots/farmland2833.JPG)</br>
![farmland4662](screenshots/farmland4662.JPG)</br>
5. Road
![road1003](screenshots/road1003.JPG)</br>
![road4298](screenshots/road4298.JPG)</br>
6. Vegetation
![vegetation3882](screenshots/vegetation3882.JPG)</br>
![vegetation4799](screenshots/vegetation4799.JPG)</br>
7. Water
![water4627](screenshots/water4627.JPG)</br>
![water5375](screenshots/water5375.JPG)</br>
