/*
Your program will decode a PNG file into an array and apply the box blur filter. Blurring an image 
reduces noise by taking the average RGB values around a specific pixel and setting it’s RGB to the 
mean values you’ve just calculated. This smoothens the colour across a matrix of pixels. For this 
assessment, you will use a 3x3 matrix. For example, if you have a 5x5 image such as the following (be 
aware that the coordinate values will depend on how you format your 2D array):
0,4 1,4 2,4 3,4 4,4
0,3 1,3 2,3 3,3 4,3
0,2 1,2 2,2 3,2 4,2
0,1 1,1 2,1 3,1 4,1
0,0 1,0 2,0 3,0 4,0

The shaded region above represents the pixel we want to blur, in this case, we are focusing on pixel 
1,2 (x,y) (Centre of the matrix). to apply the blur for this pixel, you would sum all the Red values from 
the surrounding coordinates including 1,2 (total of 9 R values) and find the average (divide by 9). This 
is now the new Red value for coordinate 1,2. You must then repeat this for Green and Blue values. 
This must be repeated throughout the image. If you are working on a pixel which is not fully 
surrounded by pixels (8 pixels), you must take the average of however many neighbouring pixels 
there are. 
NOTE – this program should work with any amount of threads.
Reading in an image file into a single or 2D array (10 marks)
Applying Box filter on image (20 marks)
Using multithreading appropriately to apply Box filter (40 marks)
Using dynamic memory – malloc (10 marks)
Outputting the correct image with Box Blur applied (20 marks)
*/