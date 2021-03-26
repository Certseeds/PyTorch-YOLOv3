#!/bin/bash
set pipefail
pip3_install(){
  pip3 install opencv-python opencv-contrib-python
  pip3 install terminaltables
  pip3 install imgaug
}
conda_install(){

}
pip3_install