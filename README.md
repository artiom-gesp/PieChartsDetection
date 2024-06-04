# Data analysis

After investigating the data, we decided to create a computer vision algorithm to tackle the challenge of cleaning the data and getting it ready for training a model. After previewing a few sets of images from the dataset, we thought we could benefit greatly by being able to separate the actual pie charts from the useless rest of the image data, for example tables and labels.

Hoping to create something stable and interpretable, we turned to computer vision. After common techniques like the Hough circle transform failed us, we chose to pursue creating our own iterative algorithm for circle detection - we first create circles with large tolerance for error all over the image, keep the most promising ones, create copies of them with less tolerance, and repeat multiple times until we converge to a stable point. To score the quality of a circle, we observe how much it overlaps with image edges, hoping to find ones where they align along the entire circumference - this metric is extremely valuable as it allows us to judge how much we have converged. If we find the largest matching circle inside an image, we can be nearly certain that this is the pie chart, similarly, we can realize when we have ended up with an unsatisfactory solution. A preview of our algorithm is the following:

![edge detection](report/circle_cv_edges.png)
![segmented circle](report/circle_cv_result.png)


By adding a fail check and running a slower and more robust version of the same method again when we fail already allows us to correctly identify more than 95% of the pie charts with accuracy being within a few pixels, but we still have problems with exploding pie charts, charts where half is missing, and charts covering only half a circle, like the two below:

![chart](report/chart_690.png)

![chart](report/chart_27849.png)

Cases like these can be simply observed by monitoring the images on which our previous algorithm fails - and here, we make a similar observation, mainly that we are interested in areas surrounded by a long circular arc. Hence we create another fallback variant to our previous algorithm - if there are no circles found in the image, we search for arcs. This has enabled us to detect the outer edges of all the aforementioned types of plot charts:

![convergence animation](report/circle_partial_anim.gif)

We use this technique to preprocess the entire dataset, replacing all pixels outside of the pie charts with black ones, before we proceed to model training. In the end, the only images we have problems segmenting are the ones with extremely small charts, plus ocasionally with ones containing a huge amount of text.

![failure case](report/failure.png)

## Percentage generation algorithm

In this section, we will describe how we transform the output of our model into the percentages required for evaluation. As described previously, our model outputs a 3-element probability distribution for each pixel - 1 layer for the background, another one for the pie chart center, and finally one for the segment boundary ends.

We start by identifying the center - we take the 2D image the model outputs and remove all points above a certain threshold. We consider their positions in the 2D space and average their positions - this will give us our estimate of the pie chart center. We continue by finding key points on the pie chart outer edge - we take the texture from our model, once again keep all points above a threshold. After that we apply a simplified version of the non-max suppression operation, merging all points that are close enough to one another.

When we have the point and the center, we simply start with a zero angle at the top of the circle. We continue clockwise, and if at any time we find a boundary keypoint, we save the angle. Afterwards we assign the percentages to be directy proportional to the angle distances between consecutive points. If for whatever reason our model fails to assign a percentage to a pie chart, leaving it empty or with one element only, we assign the percentage to one hundred.
