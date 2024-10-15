java c
CS7643: Deep Learning
Assignment 3
Deadline: Oct 14 (Oct 16 with grace period)
•  This assignment is due on the date and time posted on  Canvas.   We  will have a 48-hour grace period for this assignment.  However, no questions regarding the assignment are answered during the grace period in any form.
•  Discussion is encouraged, but each student must write his/her own answers and explicitly mention any collaborators.
•  Each student is expected to respect and follow the GT Honor Code.   We  will apply anti-cheating software to check for plagiarism. Anyone who is flagged by the software will automatically receive 0 for the homework and be reported to OSI.
•  Please do not change the filenames and function definitions in the skeleton code provided, as this will cause the test scripts to fail and you will receive no points on those failed tests.  You may also NOT change the import modules in each file or import additional modules that are not native Python packages.
•  It is your responsibility to make sure that all code and other deliverables are in the correct format and that your submission compiles and runs. We will not manually check your code (this is not feasible given the class size).  Thus, non-runnable code in our test environment will directly lead to a score of 0.  Also, be sure to clean up print statements, etc. before submitting – the autograder will likely reject your entire submission and we would not be able to grant you any points.
...there is still little insight into the internal operation and behavior. of these complex models, or how they achieve such good performance. From a scientific standpoint, this is deeply unsatisfactory. With-out clear understanding of how and why they work, the development of better models is reduced to trial-and-error.
Zeiler and Fergus.  2014
Interpretability matters. In order to  build trust in intelligent systems and move towards their meaningful integration into our everyday lives, it is clear that we must build ‘transparent’ models that have the ability to explain why they predict what they predict.Selvaraju et al.  2016
OverviewThis assignment has two main parts.  The first part explores the use of different types of saliency methods, which provide insight into the decision-making processes of image convolutional networks (CNNs).  The second part involves using gradient manipulation techniques to extract the content and style. from different images and combine them to produce creative works.  If you are unfamiliar with these concepts, review relevant course material and the required readings for this assignment.  The concepts covered will provide you with valuable tools for model analysis, including explainability, bias determinations, and debugging, which can be extended to other domains.
For this assignment, we will use conda to create a Python environment with the required packages installed.Please note that our environment does NOT have PyTorch Installation for you be- cause you may use CPU/GPU or a different version of CUDA. To install PyTorch, please refer to the official documentation and select the options based on your local OS. We recommend using PyTorch 1.3 or 1.4 to finish the problems in this assignment, which has been tested with Python3.7 on Linux and Mac although PyTorch versions ≥ 1.0 should generally work as well.
1       conda  env  create  -f  environment.yaml
2        conda  activate  cs7643-a3
DeliverablesYou must submit your write-up and solution code to Gradescope by the assigned due date and time. The Gradescope Autograder will test your submissions, but keep in mind that the feedback provided is limited.  To ensure your code is functioning properly, you should write your own tests and assertions. We also provide samples of the deliverables for you to check against your work.Note:  The code you submit should be vectorized wherever possible.   Code  that  uses non-vectorized constructs like for-loops may not pass the Autograder, particularly when executing the forward pass for multiple images.To submit your code to Gradescope, you will need to upload a zipped file containing all your code with the folder structure intact.  You can run collect  submission.py to auto- mate this process.  Once complete, upload assignment   3   submission.zip to Gradescope Assignment 3 Code.Note:  Although passing Gradescope tests is important, it does not guarantee that your code is free of bugs.  An image that looks similar to a sample image may still be the result of a flawed implementation.  To avoid this, diligently follow the recipes outlined in the skeleton code and referenced papers.  Keep in mind that images output by your work will show minor random run-to-run variations, which is expected and should not be altered.For your write-up, a report template called report-template-a3.pptx has been pro- vided in the root directory (./).  Please follow the instructions for each question carefully and be sure to include your visualizations and stylized images.  To help manage your time effectively, we recommend allocating a maximum of 350 words per answer, unless other- wise specified. Points may be deducted for incorrect tagging or if your report exceeds the maximum word count.
Once you have completed your write-up, upload it to Gradescope Assignment 3  Written, ensuring that you assign the correct pages to the corresponding questions.
1   Part I: Network Visual InspectionsIn the  first  section  of Part  I,  we  will  apply  gradient  and  signal  methods  on  top  of SqueezeNet, a compact CNN model that achieves high performance on the ImageNet dataset while being significantly smaller than other models, such as VGG and ResNet. SqueezeNet has a file size of less than 5 megabytes, making it well-suited for deployment on memory-limited devices.  According to the original paper, SqueezeNet achieves a top-1 accuracy of  60% and a top-5 accuracy of  80% on the ImageNet dataset.
In this section, we will implement the following techniques:
•  Class Model Visualizations: We will synthesize an image to maximize the classi- fication score of a particular class to provide insights into what the network focuses on when classifying images of that class.
•  Class-Specific  Saliency  Maps  for  Images:   We  will  generate  image-specific saliency maps to quickly determine which parts of an image influenced the network’s classification decision for a particular class.
•  Fooling Images: We can perturb an input image so that it appears the same to humans, but will be misclassified by the pretrained network.
•  GradCAM: We will use Gradient Class Activation Mapping (GradCAM) to high- light the areas in an image that are most relevant to a given label.
1.1   Saliency mapUsing this pretrained model, we will compute class saliency maps as described in the paper:  [1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep Inside Con- volutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014.A saliency map tells us the degree to which each pixel in a given image affects the classifi- cation score for that image.  To compute it, we compute the gradient of the unnormalized score corresponding to the correct class (which is a scalar) with respect to the pixels of the image.  If the image has shape (3, H, W), then this gradient will also have shape (3, H, W); for each pixel in the image, this gradient tells us the amount by which the classification score will change if the pixel changes by a small amount at that pixel.  To compute the saliency map, we take the absolute value of this gradient, then take the maximum value over the 3 input channels element-wise; the final saliency map thus has shape (H, W), and all entries are non-negative.
Your tasks are as follows:
1.  Follow the instructions in  .visualizers/saliency_map.py and implement func- tions to manually compute saliency maps.
2.  Follow instructions and implement Saliency Map with Captum in root/saliency_map.py
3. As the final step, run root/saliency_map.py which will run both the manual and captum computations and output images for visualization.
1.2   GradCam
GradCAM (which stands for Gradient Class Activation Mapping) is a technique that tells
us where a convolutional network is looking when it is making a decision on a given input image. There are three main stages to it:
1.  Guided Backprop (Changing ReLU Backprop Layer, Link)
2.  GradCAM (Manipulating gradients at the last convolutional layer, Link)
3.  Guided GradCAM (Pointwise multiplication of above stages)
In this section, you will be implementing these three stages to recreate the full GradCAM pipeline. Your tasks are as follows:
1.  Follow instructions and implement functions in visualizers/gradcam.py, which manually computes guided backprop and GradCam
2.  Follow instructions and implement GradCam with Captum in root/gradcam.py
3. As the final step, run root/gradcam.py which will run both the manual and captum computations and output images for visualization.
1.3   Fooling ImageWe can also use the similar concept of image gradients to study the stability of the network.   Consider  a state-of-the-art deep neural network that generalizes well on an object recognition ta代 写CS7643: Deep Learning Assignment 3Python
代做程序编程语言sk.  We expect such network to be robust to small perturbations of its input, because small perturbation cannot change the object category of an image. However, [2] find that applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction.
[2] Szegedy et al, ”Intriguing properties of neural networks”, ICLR 2014. linkGiven an image and a target class, we can perform. gradient ascent over the image to maximize the target class, stopping when the network classifies the image as the target class. We term the so perturbed examples “adversarial examples”.
Read the paper, and then implement the following function to generate fooling images. Your tasks are as follows:
1.  Follow instructions and implement functions in visualizers/fooling_image.py, which manually computes the fooling image
2. As the final step, you should run the corresponding section in root/fooling_image.py to generate fooling images.
1.4   Class Model VisualizationsFirst up are class model visualizations. This idea was first presented by Simonyan et al. and later extended by Yosinski et al.  to include regularization techniques that improve the quality of the generated image.Concretely, let I be an image and let y be a target class.  Let sy (I) be the score that a convolutional network assigns to the image  I  for  class  y;  note  that  these  are  raw unnormalized scores,  not  class  probabilities.   We  wish  to  generate  an  image  I*   that achieves a high score for the class y by solving the problem
where R is a (possibly implicit) regularizer (note the sign of R(I) in the argmax:  we want to minimize this regularization term).   We  can solve this optimization problem using gradient ascent, computing gradients with respect to the generated image.  We will use L2 regularization (squared L2 norm) of the form.
and implicit regularization (as suggested by Yosinki et al.) by periodically blurring the generated image.   We  can  solve  this  problem using gradient ascent on the generated image.
Your tasks are as follows:
1.  Follow instructions and implement functions in visualizers/class_visualization.py, which manually computes the class visualization
2.  As the final step, you should run the corresponding section in root/class_visualization.py to generate the visualizations.
2   Part II: Style TransferAnother task closely related to image gradients is style. transfer.   Style. transfer  is  a technique that allows us to apply the style. of one image to the content of another, resulting in a new image that combines the two.  This technique has become increasingly popular in computer vision and deep learning, as it allows us to generate blended images that combine the content of one image with the style. of another.  We will study and implement the style. transfer technique from:
•  Gatys et al., ”Image Style. Transfer Using Convolutional Neural Networks”, CVPR
2015. paper link
The general idea is to take two images (a content image and a style. image), and produce a new image that reflects the content of one but the artistic ”style” of the other. We will do this by first formulating a loss function that matches the content and style. of each respective image in the feature space of a deep network, and then performing gradient descent on the pixels of the image itself.In this assignment, we will also use SqueezeNet as our feature extractor which can easily work on a CPU bound machine. Similarly, if computational resources are not any problem for you, a larger network which may enhance the visual outputs.
2.1   Content LossWe can generate an image that reflects the content of one image and the style. of another by incorporating both in our loss function.  We want to penalize deviations from the content of the content image and deviations from the style. of the style. image.  We can then use this hybrid loss function to perform. gradient descent not on the parameters of the model, but instead on the pixel values of our original image.
Let’s first write the content loss function.  Content loss measures how much the featuremap of the generated image differs from the feature map of the source image.  We only care about the content representation of one layer of the network (say, layer l), that has feature maps Al   ∈ R1×Cl×Hl×Wl .  Cl  is the number of channels in layer l, Hl  and Wl are the height and width.  We will work with reshaped versions of these feature maps that combine all spatial positions into one dimension.  Let Fl   ∈ RNl×Ml   be the feature map for the current image and Pl  ∈ RNl×Ml    be the feature map for the content source image where Ml  = Hl  × Wl  is the number of elements in each feature map.  Each row of Fl  or Pl  represents the vectorized activations of a particular filter, convolved over all positions of the image.  Finally, let wc  be the weight of the content loss term in the loss function.
Then the content loss is given by:

1.  Implement Content Loss in style_modules/content_loss.py
You can check your implementation by running the ’Test content loss’ function.  The expected error should be 0.0
2.2   Style Loss
Now we can tackle the style. loss.  For a given layer l, the style. loss is defined as fol- lows:First, compute the Gram matrix G which represents the correlations between the re- sponses of each filter, where F is as above. The Gram matrix is an approximation to the covariance matrix – we want the activation statistics of our generated image to match the activation statistics of our style. image, and matching the (approximate) covariance is one way to do that.  There are a variety of ways you could do this, but the Gram matrix is nice because it’s easy to compute and in practice shows good results.
Given a feature map Fl  of shape (1, Cl , Ml ), the Gram matrix has shape (1, Cl , Cl ) and its elements are given by:
Assuming Gl  is the Gram matrix from the feature map of the current image, Al  is the Gram Matrix from the feature map of the source style. image, and wl  a scalar weight term, then the style. loss for the layer l is simply the weighted Euclidean distance between the two Gram matrices:

In practice we usually compute the style. loss at a set of layers L rather than just a single layer l; then the total style. loss is the sum of style. losses at each layer:

1.  Implement Style. Loss in style_modules/style_loss.py
You can check your implementation by running the ’Test style. loss’ function.  The ex-
pected error should be 0.0
2.3   Total Variation LossIt turns out that it’s helpful to also encourage smoothness in the image. We can do this by adding another term to our loss that penalizes wiggles or total variation in the pixel values.  This concept is widely used in many computer vision task as a regularization term.

You can compute the total variation as the sum of the squares of differences in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically).  Here we sum the total-variation regularization for each of the 3 input channels (RGB), and weight the total summed loss by the total variation weight, wt :

You may not see this loss function referenced, but you should be able to implement it based on this equation.
You should try to provide an efficient vectorized implementation.
1.  Implement Style. Loss in style_modules/tv_loss.py
You can check your implementation by running ’Test total variation loss’ function.  The expected error should be 0.0
2.4   Style TransferYou have implemented all the loss functions in the paper.  Now we’re ready to string it all together.  Please read the entire function:  figure out what are all the parameters, inputs, solvers, etc.  The update rule in function style   transfer of style    utils.py is held out for you to finish.
As the final step, run the script. style   transfer.py to generate stylized images.
2.5   Style Transfer - Unleash Your CreativityYou now have the structure built to transfer style. from one image to another.  For this section, select two images of your choosing - a content image and a style. image - from any non-copyrighted source.  Now affect a style. transfer to generate a stylized image.  Note that some image preprocessing may be required before your selected images are input into the code you developed above. You can examine images provided by us for the previous section to get an idea what may be required.
Include  your  two  selected  images  (before)  and  the  stylized  image  (after)  in  the  re- port.
3   Wrap-upFinally, choose one of the papers below to read and analyze with respect to this assign- ment. Then provide a short summary regarding the papers main contributions, followed by your observations and personal takeaways.
1.  Mukund Sundararajan, Ankur Taly, Qiqi Yan,  ”Axiomatic Attribution for Deep Networks”, ICML 2017
2.  Julius Adebayo,  Justin  Gilmer,  Michael  Muelly,  Ian Goodfellow,  Moritz Hardt, Been Kim, ”Sanity Checks for Saliency Maps”, arXiv 2018
3.  Quan Zheng, Ziwei Wang, Jie Zhou, Jiwen Lu.  ”Shap-CAM: Visual Explanations for Convolutional Neural Networks based on Shapley Value” arXiv 2022




         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
