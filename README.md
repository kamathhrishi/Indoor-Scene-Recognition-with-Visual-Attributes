# Indoor-Scene-Recognition-with-Visual-Attributes

<p align="justify">In this project , I trained a classifier to recognize 67 different categories of Indoor Scenes from the MIT Indoor 67 dataset. The dataset is a small dataset with just 5400 images for training , so a classifier trained from scratch ended up overfitting to an large extent giving an very low accuracy of ~35%. This was obtained after several stratergies. But , using visual attributes from the SUN dataset with attributes , the classifier accuracy can go upto 60%. This does not even require learning representations from a large dataset such as SUN which consists of 2 million images with indoor/outdoor scenes. But , annotating attributes is definitely expensive. </p>

<p align="justify">The project consists of two networks , a attribute prediction network and scene recognition network. The attribute prediction network is trained on SUN attribute dataset and the scene detector network combines predicted attribute along with features extracted from a given image at the linear layer to make predictions.</p>

<center>
<figure>
<img src="http://web.mit.edu/torralba/www/allIndoors.jpg"></img>
   <figcaption><b>MIT 67 Indoor Dataset</b></figcaption>
</figure>
</center>

<figure>
<img src="http://cs.brown.edu/~gmpatter/website_imgs/pca2D_w_nn.jpg"></img>
<figcaption><b>SUN Attribute Dataset</b></figcaption>
</figure>


<h2>Instructions</h2>

Setup required packages 

```pip3 install -r requirements.txt```
   
in the root directory of the project 

<b>Attribute Network</b>

1. Download SUN Dataset with attributes
   Download Images: http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz and unzip it
   and place the folder  in directory as Attribute_Network. 
   Download Attribute Dataset: http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz and unzip it 
   and place folder in the directory as "Scene Detector Network"
   
2. Run the Attribute Network Training script by running 

   ``` python3 main.py ```
   
3. Upon training , the trained model is stored as model.pth in the Attribute Network directory which predicts
   visual attributes for training the Indoor Scene Recognition Network
   
   
<b>Scene Recognition Network</b>

1. Download MIT Indoor 67 dataset and place train and test folder in the scene Detector Network
   directory.
   
2. Run the scene detector training scripts by running

   ```python3 main.py```

3. After completion of training , the model is stored in the Scene detector network directory as 
   model.pth
