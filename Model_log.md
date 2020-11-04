
##[Based network]
### Pytorch VGG19

[Lastest update] : 2020.11.03


**[Dataset]**

v1.x English single character set(external data)    
v2.x skeletonized external data     
v3.x Skeletonized_character_dataset6  : Recognition character ->  dataset   
v4.x EMNIST OCR 62 label dataset *      

ver 6.0x Simple network with original skeletonized dataset (crop from CRAFT, Deep-text data)    
ver 6.1x Simple network with generated synth char dataset   

ver 7.0x Decision Tree  

-----------------------------------------------------

2020.09.28 mon

Data rebuliding...
version sequence also change..








###[VGG19 version]

ver 1.0 batch 8, epoch 5    
ver 1.1 batch 8, epoch 10   
ver 1.2 batch 8, epoch 20   
ver 1.3 batch 4, epoch 5    

ver 2.1 batch 8 epoch 5, skeletonize(external data) 
ver 2.2 batch 8 epoch 10, skeletonize(external data)    
ver 2.3 batch 4, epoch 10, skeletonize(external data)   
ver 2.4 batch 4, epoch 5, skeletonize(external data), change f1,f2,f3 layers    

ver 4.0 batch 8 epoch 10 VGG 26+26+10 case,digit of EMNIST dataset. 
ver 4.1 batch 16 epoch 10   
ver 4.2 batch 4  epoch 10 , resize 224 -> 28*28 784 
ver 4.3 batch 4 epoch 10(test), resize244 to 784, working fc layer  
ver 4.4 batch 4 epoch 20 resize 784, fc3    
ver 4.5 batch 4 epoch 100 resize 224, fc3 -> to late    
ver 4.6 batch 4 epoch 10 resize 224 balanced    
ver 4.7 batch 4 epoch 30 resize 224 balanced    

ver 5.3 VGG19 , batch 16, epoch 10 resize 224, with EMNIST balance  

[EMNIST_Letter_vgg and spinalVGG.py]xxxxxxxxxxxxxx

ver 5.0 spinalnet + vgg5 with EMNIST byclass    

ver 5.1 spinalnet + vgg5 with EMNIST balance     Valid : 90 | test : 24 

================================================

###[ConnNet]

ver 6.x Simple network


ver 7.x Decision Tree




## ----------Final model--------------

###[VGG19]

ver 5.3 VGG19 , batch 16, epoch 10 resize 224, with EMNIST balance      
[Dataset: balanced] : Valid : 90% | test : 88% : 

ver 4.5 batch 4 epoch 100 resize 224, fc3, with EMNIST byclass          
[Dataset: byclass] Valid : 77% | test : 76%     # resize 784 /784

* Need rename with model.
convnet -> connNet, version start with 6.x

Old
ConnNet_v1.2+VGG19_v5.3 
->
New
ConnNet_v6.x_+_OCR_v5.x_ep00_batch00_





===============================================================================================
