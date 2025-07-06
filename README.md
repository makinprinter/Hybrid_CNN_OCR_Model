This is a hybrid CNN and rules-based model I created as a novel approach to OCR.
It was not designed to be a practical replacement for any current OCR program but rather as a test of a hybrid rules-based approach.

The way my hybrid system works is that it sorts images of letters based on whether they contain vertical lines, horizontal lines, or closed circles. It then sends the image to a custom CNN model that is trained only on images with that specific set of attributes.

The following is a diagram of how the system works:



                [Image]
                   ||
                   \/
	  [Rules-Based Sorting Model]
            /       |       \
	[model_v] [model_h] [model_o]
                              |
                       Image is a "Q"



 
Example:
The input image is the letter "Q".

Why This Approach?

The goal of this experiment was to explore whether combining classical image processing techniques (feature detection) with deep learning (CNNs) can improve performance—especially in scenarios where:

Certain character groups share distinct visual traits,

Training data can be subdivided into more manageable, specialized pools,

Model interpretability or modularity is a focus.

By breaking down the OCR problem into sub-problems, each CNN can specialize, which might reduce confusion between visually similar characters within the same attribute group.


