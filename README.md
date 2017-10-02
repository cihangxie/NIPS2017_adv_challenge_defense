# NIPS2017_adv_challenge_defense
Utilize randomization to defend adversarial examples. 
Challenge URL: https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack

## The approach
The main ideal of the defense is to utilize randomization to defend adversarial examples:
- Resizing: after pre-processing, resize the original image (size of 299 x 299 x 3) to a larger size, Rnd x Rnd x 3,  randomly, where Rnd is within range [299, 331). 
- Padding: after resizing, pad the resized image to a new image with size 331 x 331 x 3, where the padding size at left, right, upper, bottom are [a, 331-Rnd-a, b, 331-Rnd-b]. The possible padding pattern for the size Rnd is (331-Rnd+1)^2.

Combining these two randomization methods toghther, we can create 12528 different patterns in total. 

## The motivation
1. Resizing changes images pixel values, which means it affects CNN low-level feature extraction.
2. Padding changes the shape CNN feature, which means it affects CNN high-level feature classfication. 
3. Deep networks are robust to resizing and padding operations when images are not adversarial.

## Pros compared to other defending methods
1. No training is required
2. Compatiable to other defending methods (i.e., we use randomization + adversarial training in our submission)
3. Able to mitigate white-box adversarial attack

## Models
- http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz


