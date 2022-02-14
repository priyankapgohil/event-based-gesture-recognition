# Event based Gesture Recognition

Gesture recognition on Event Data usually involves computation of features using either Bag of Visual Words, image/scene reconstruction or other expensive aggregation techniques that lose the Spatio-temporal information associated with the stream of event data. 

In this project, we propose a gesture recognition system that involves computing <strong>x - y, y - t and x - t motion maps</strong> of event camera data to feed as input images to a convolutional network. 

We introduce the <strong>Stack of ResNets Model</strong> and <strong>9 Channel ResNet model</strong>
as suitable convolutional network architectures to use in combination with the motion map inputs. We apply our strategy to the IITM DVS 10 Gesture Dataset and show that our model obtains the state of the art results.

# Credits

Thanks to S. A. Baby at al for the IITM DVS128 Gesture dataset and motion-maps code -
https://github.com/Computational-Imaging-Lab-IITM/HAR-DVS 
