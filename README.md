This is a Hybrid CNN and rules based model. I created as a novel approach to OCR. 
It was not created to be a practical replacement for any current OCR program but as a test of a hybrid-rules approach. 
The way my hybrid system works is that it sorts images of letters based if there is vertical lines, horizonal lines or closed circle's in the image. It then sends the image to a custom CNN model that is only trained on images with that images set of attributes. 

The following is a diagram of how the system works. 



	       [image]
	          ||
	          \/

	  [Rules based Sorting model] # It chooses O in this is case. In addition, the full system connects to model_v_h, model_h_o and so on.  
	      /      |       \
 [model_v] [model_H] [model_O]# each model is a conventional neural network. 
                         | 
		             	image is a "Q"


 
example image is a "Q". 


Why This Approach?
The goal of this experiment was to explore whether combining classical image processing techniques (feature detection) with deep learning (CNNs) can improve performance, especially in scenarios where:

Certain character groups share distinct visual traits,

Training data can be subdivided into more manageable, specialized pools,

Model interpretability or modularity is a focus.

By breaking down the OCR problem into sub-problems, each CNN can specialize, which might reduce confusion between visually similar characters in the same attribute group.
