# YoloFromScratch

Steps to run the script:
- Pip install all the dependencies using:

  ```
  pip install -r yolofromscratch/requirements.txt
  ```
- The dataset should be in the following tree structure:

### yolofromscratch
  - Project
    - data
      - images
        - train
          - xxx.jpg
          - yyy.jpg
        - valid
          - zzz.jpg
      - labels
        - train
          - xxx.txt
          - yyy.txt
        - valid
          - zzz.txt


- After the dataset has been formatted in the proper way (above tree structure), run the ``train.py`` script:
  ```
  python yolofromscratch/train.py
  ```

  If everything is correct, the training will begin for 50 epochs with a batch size of 8. Following output will be observed:

  ![image](https://github.com/HamzaSulaiman1996/YoloFromScratch/assets/109808023/546648a6-dce7-494f-b631-61393193dec9)


  > Note: The purpose of this project was to dive deeper into the detailed working of Yolo Algorithm and to have a better understanding of the principles involved.
  > This implementation of yolo is correlated with the original yolov1 paper. However, resnet-18 architecture is used as the backbone for classification. The reason is that the Yolo architecture described in the paper was too big to fit in my GPU memory. However, the Yolo architecture is also implemented in the ``model.py`` script with its own class ``Yolov1`` 


  
