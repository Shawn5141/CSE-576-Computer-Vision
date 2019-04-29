#include <iostream>
#include <assert.h>

#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>

/***********************************************************************
  This is the only file you need to change for your assignment. The
  other files control the UI (in case you want to make changes.)
************************************************************************/

/***********************************************************************
  The first eight functions provide example code to help get you started
************************************************************************/


// Convert an image to grayscale
void MainWindow::BlackWhiteImage(QImage *image)
{
    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int noiseMag = mag*2;

    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            int red = qRed(pixel), green = qGreen(pixel), blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;
                red += noise; green += noise; blue += noise;
            }
            image->setPixel(c, r, qRgb(max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))));
        }
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    int w = image.width();
    int h = image.height();
    QImage buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(int r=0;r<h/2;r++)
        for(int c=0;c<w/2;c++)
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
}

// Round float values to the nearest integer values and make sure the value lies in the range [0,255]
QRgb restrictColor(double red, double green, double blue)
{
    int r = (int)(floor(red+0.5));
    int g = (int)(floor(green+0.5));
    int b = (int)(floor(blue+0.5));

    return qRgb(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)));
}

// Normalize the values of the kernel to sum-to-one
void NormalizeKernel(double *kernel, int kernelWidth, int kernelHeight)
{
    double denom = 0.000001; int i;
    for(i=0; i<kernelWidth*kernelHeight; i++)
        denom += kernel[i];
    for(i=0; i<kernelWidth*kernelHeight; i++)
        kernel[i] /= denom;
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it's not done here.
// Note: This function is written using QImage form of the input image. But all other functions later use the double form
void MainWindow::MeanBlurImage(QImage *image, int radius)
{

    if(radius == 0)
        return;
    int size = 2*radius + 1; // This is the size of the kernel


    //**********************************//
    std::cout<<"size :"<<size<<std::endl;





    // Note: You can access the width and height using 'imageWidth' and 'imageHeight' respectively in the functions you write
    int w = image->width();
    int h = image->height();

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders (zero-padding)
    QImage buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute the kernel to convolve with the image
    double *kernel = new double [size*size];

    for(int i=0;i<size*size;i++)
        kernel[i] = 1.0;

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // For each pixel in the image...
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double rgb[3];
            rgb[0] = rgb[1] = rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(int rd=-radius;rd<=radius;rd++)
                for(int cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     //For the functions you write, check the ConvertQImage2Double function to see how to get the pixel value
                     QRgb pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
            // Store the pixel in the image to be returned
            // You need to store the RGB values in the double form of the image
            image->setPixel(c, r, restrictColor(rgb[0],rgb[1],rgb[2]));
        }
    }
    // Clean up (use this carefully)
    delete[] kernel;
}

// Convert QImage to a matrix of size (imageWidth*imageHeight)*3 having double values
void MainWindow::ConvertQImage2Double(QImage image)
{
    // Global variables to access image width and height
    imageWidth = image.width();
    imageHeight = image.height();


    /**********************************/
    std::cout<<"W: "<<imageWidth<<" h: "<<imageHeight<<"\n"<<endl;

    // Initialize the global matrix holding the image
    // This is how you will be creating a copy of the original image inside a function
    // Note: 'Image' is of type 'double**' and is declared in the header file (hence global variable)
    // So, when you create a copy (say buffer), write "double** buffer = new double ....."
    Image = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
            Image[i] = new double[3];

    // For each pixel
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
        {
            // Get a pixel from the QImage form of the image
            QRgb pixel = image.pixel(c,r);

            // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
            Image[r*imageWidth+c][0] = (double) qRed(pixel);
            Image[r*imageWidth+c][1] = (double) qGreen(pixel);
            Image[r*imageWidth+c][2] = (double) qBlue(pixel);
        }
}

// Convert the matrix form of the image back to QImage for display
void MainWindow::ConvertDouble2QImage(QImage *image)
{
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
            image->setPixel(c, r, restrictColor(Image[r*imageWidth+c][0], Image[r*imageWidth+c][1], Image[r*imageWidth+c][2]));
}


/**************************************************
 TIME TO WRITE CODE
**************************************************/

/**************************************************
 TASK 1
**************************************************/

// Convolve the image with the kernel
void MainWindow::Convolution(double** image, double *kernel, int kernelWidth, int kernelHeight, bool add)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * kernel: 1-D array of kernel values
 * kernelWidth: width of the kernel
 * kernelHeight: height of the kernel
 * add: a boolean variable (taking values true or false)
*/
{
    // Add your code here

    int buffer_imageWidth=imageWidth+kernelWidth-1;   //Should I minus 1
    int buffer_imageHeight=imageHeight+kernelHeight-1;


    //Create a buffer with kernel size added
    int buffer_size=buffer_imageHeight*buffer_imageWidth;
    double **buffer = new double *[buffer_imageHeight*buffer_imageWidth] ;
    for (int M=0;M<buffer_size;M++){
        buffer[M]=new double [3];
    }


    //Normalized Kernel
    //NormalizeKernel(kernel, kernelWidth, kernelHeight);

    // For each pixel

    for (int r = 0; r < buffer_imageHeight; r++){
        for (int c = 0; c < buffer_imageWidth; c++)
        {
            // Get a pixel from the QImage form of the
            if (r< int(kernelHeight/2) || r>=imageHeight+int(kernelHeight/2) || (c<int(kernelWidth/2) || c>=imageWidth+int(kernelWidth/2))){
                buffer[r*buffer_imageWidth+c][0]=0.0 ;
                buffer[r*buffer_imageWidth+c][1]=0.0 ;
                buffer[r*buffer_imageWidth+c][2]=0.0 ;
                //std::cout<<buffer[r*buffer_imageWidth+c][0];
             }
            else{
                int _r= r-int(kernelHeight/2);
                int _c= c-int(kernelWidth/2);

                 // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
                 //std::cout<<r*buffer_imageWidth+c<<std::endl;
                 buffer[r*buffer_imageWidth+c][0]=Image[_r*imageWidth+_c][0] ;
                 buffer[r*buffer_imageWidth+c][1]=Image[_r*imageWidth+_c][1] ;
                 buffer[r*buffer_imageWidth+c][2]=Image[_r*imageWidth+_c][2] ;


                 //std::cout<<buffer[r*buffer_imageWidth+c][0]<<" ";
            }


         }
        //std::cout<<"\n"<<std::endl;

}




    for (int r = 0; r < imageHeight; r++){
        for (int c = 0; c < imageWidth; c++){
            double value[3] = {0.0,0.0,0.0};

            for(int rd=-int(kernelHeight/2);rd<=int(kernelHeight/2);rd++){
                for(int cd=-int(kernelWidth/2);cd<=int(kernelWidth/2);cd++){

                    // Get the value of the kernel
                    double w = kernel[kernelWidth*(rd+int(kernelWidth/2))+(cd+int(kernelHeight/2))];

                    int index_r=r+int(kernelHeight/2)+rd;
                    int index_c=c+int(kernelWidth/2)+cd;
                    //std::cout<<"total "<<buffer[0][0]<<std::endl;

                    if(buffer_imageHeight*buffer_imageWidth<index_r*buffer_imageWidth+index_c){
                        std::cout<<"Fuck you"<<std::endl;
                    }
                    //add weight after test
                    value[0]+=buffer[index_r*buffer_imageWidth+index_c][0]*w;
                    value[1]+=buffer[index_r*buffer_imageWidth+index_c][1]*w;
                    value[2]+=buffer[index_r*buffer_imageWidth+index_c][2]*w;
                }
            }
            // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
            if (add==true){
                Image[r*imageWidth+c][0]=value[0]+128.0;
                Image[r*imageWidth+c][1]=value[1]+128.0;
                Image[r*imageWidth+c][2]=value[2]+128.0;
            }
            else{
                Image[r*imageWidth+c][0]=value[0];
                Image[r*imageWidth+c][1]=value[1];
                Image[r*imageWidth+c][2]=value[2];
            }
            std::cout<<Image[r*imageWidth+c][0]<<" ";
        }
        std::cout<<"\n"<<std::endl;
    }




std::cout<<"Convolution fininished"<<std::endl;

delete [] kernel;

for (int l=0;l<buffer_size;l++)
    delete [] buffer[l];
delete [] buffer;




}

/**************************************************
 TASK 2
**************************************************/

// Apply the 2-D Gaussian kernel on an image to blur it
void MainWindow::GaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    double radius = 3*sigma;
    int kernelWidth =int(2*radius)+1;
    int kernelHeight =int(2*radius)+1;
    bool add=false;
    double *kernel =new double [kernelWidth*kernelHeight];
    for(int x=0;x<kernelHeight;x++){
        for(int y=0;y<kernelWidth;y++){
         kernel[x*kernelHeight+y]= (exp(-(x*x+y*y)/2/(sigma*sigma)))/2/M_PI/(sigma*sigma);
         //std::cout<<kernel[x*kernelHeight+y]<<" ";
        }
    //std::cout<<" "<<std::endl;
    }

    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);



}

/**************************************************
 TASK 3
**************************************************/

// Perform the Gaussian Blur first in the horizontal direction and then in the vertical direction
void MainWindow::SeparableGaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    double radius = 3*sigma;
    int kernelWidth =int(2*radius)+1;
    int kernelHeight =int(2*radius)+1;
    bool add=false;
    double *kernel =new double [kernelWidth*kernelHeight];

    //Horizontal kernel
    for(int x=0;x<kernelHeight;x++){
        for(int y=0;y<kernelWidth;y++){
            if (x==int(kernelHeight/2)){
                kernel[x*kernelHeight+y]= (exp(-(x*x+y*y)/2/(sigma*sigma)))/2/M_PI/(sigma*sigma);
            }else{
                kernel[x*kernelHeight+y]=0.0;
            }
        }
    }
    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);

    //Vertical kernel
    for(int x=0;x<kernelHeight;x++){
        for(int y=0;y<kernelWidth;y++){
            if (y==int(kernelWidth/2)){
                kernel[x*kernelHeight+y]= (exp(-(x*x+y*y)/2/(sigma*sigma)))/2/M_PI/(sigma*sigma);
            }else{
                kernel[x*kernelHeight+y]=0.0;
            }
        }
    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);
    }
}

/********** TASK 4 (a) **********/

// Compute the First derivative of an image along the horizontal direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_x(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    /*double *kernel = new double[3];
    kernel[0]=-1.0;
    kernel[1]=0.0;
    kernel[2]=1.0;
    bool add=false;
    int kernelWidth =3;
    int kernelHeight=1;*/
    double *kernel = new double[9];
    kernel[0]=0.0;
    kernel[1]=0.0;
    kernel[2]=0.0;
    kernel[3]=-1.0;
    kernel[4]=0.0;
    kernel[5]=1.0;
    kernel[6]=0.0;
    kernel[7]=0.0;
    kernel[8]=0.0;
    bool add=true;
    int kernelWidth =3;
    int kernelHeight=3;


    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);
    GaussianBlurImage(Image,sigma);
}

/********** TASK 4 (b) **********/

// Compute the First derivative of an image along the vertical direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_y(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    // Add your code here
    double *kernel = new double[9];
    kernel[0]=0.0;
    kernel[1]=1.0;
    kernel[2]=0.0;
    kernel[3]=0.0;
    kernel[4]=0.0;
    kernel[5]=0.0;
    kernel[6]=0.0;
    kernel[7]=-1.0;
    kernel[8]=0.0;
    bool add=true;
    int kernelWidth =3;
    int kernelHeight=3;
    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);
    GaussianBlurImage(Image,sigma);
}

/********** TASK 4 (c) **********/

// Compute the Second derivative of an image using the Laplacian operator and then apply Gaussian blur
void MainWindow::SecondDerivImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    double *kernel = new double[9];
    kernel[0]=0.0;
    kernel[1]=1.0;
    kernel[2]=0.0;
    kernel[3]=1.0;
    kernel[4]=-4.0;
    kernel[5]=1.0;
    kernel[6]=0.0;
    kernel[7]=1.0;
    kernel[8]=0.0;
    bool add=true;
    int kernelWidth =3;
    int kernelHeight=3;
    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);
    GaussianBlurImage(Image,sigma);
}

/**************************************************
 TASK 5
**************************************************/

// Sharpen an image by subtracting the image's second derivative from the original image
void MainWindow::SharpenImage(double** image, double sigma, double alpha)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
 * alpha: constant by which the second derivative image is to be multiplied to before subtracting it from the original image
*/
{
    // Add your code here
    SecondDerivImage(Image,sigma);
    for (int r = 0; r < imageHeight; r++){
        for (int c = 0; c < imageWidth; c++){
            Image[r*imageWidth+c][0]*=alpha-128.0;
            Image[r*imageWidth+c][1]*=alpha-128.0;
            Image[r*imageWidth+c][2]*=alpha-128.0;

        }

    }

}

/**************************************************
 TASK 6
**************************************************/

// Display the magnitude and orientation of the edges in an image using the Sobel operator in both X and Y directions
void MainWindow::SobelImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
*/
{
    // Add your code here
    //Sobel kernel (vertical Edge)
    double *kernel = new double[9];
    kernel[0]=1.0;
    kernel[1]=0.0;
    kernel[2]=-1.0;
    kernel[3]=2.0;
    kernel[4]=0.0;
    kernel[5]=-2.0;
    kernel[6]=1.0;
    kernel[7]=0.0;
    kernel[8]=1.0;
    bool add=false;
    int kernelWidth =3;
    int kernelHeight=3;
    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);






    // Use the following 3 lines of code to set the image pixel values after computing magnitude and orientation
    // Here 'mag' is the magnitude and 'orien' is the orientation angle in radians to be computed using atan2 function
    // (sin(orien) + 1)/2 converts the sine value to the range [0,1]. Similarly for cosine.

    // image[r*imageWidth+c][0] = mag*4.0*((sin(orien) + 1.0)/2.0);
    // image[r*imageWidth+c][1] = mag*4.0*((cos(orien) + 1.0)/2.0);
    // image[r*imageWidth+c][2] = mag*4.0 - image[r*imageWidth+c][0] - image[r*imageWidth+c][1];
}

/**************************************************
 TASK 7
**************************************************/

// Compute the RGB values at a given point in an image using bilinear interpolation.
void MainWindow::BilinearInterpolation(double** image, double x, double y, double rgb[3])
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * x: x-coordinate (corresponding to columns) of the position whose RGB values are to be found
 * y: y-coordinate (corresponding to rows) of the position whose RGB values are to be found
 * rgb[3]: array where the computed RGB values are to be stored
*/
{
    // Add your code here
}

/*******************************************************************************
 Here is the code provided for rotating an image. 'orien' is in degrees.
********************************************************************************/

// Rotating an image by "orien" degrees
void MainWindow::RotateImage(double** image, double orien)

{
    double radians = -2.0*3.141*orien/360.0;

    // Make a copy of the original image and then re-initialize the original image with 0
    double** buffer = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
    {
        buffer[i] = new double [3];
        for(int j = 0; j < 3; j++)
            buffer[i][j] = image[i][j];
        image[i] = new double [3](); // re-initialize to 0
    }

    for (int r = 0; r < imageHeight; r++)
       for (int c = 0; c < imageWidth; c++)
       {
            // Rotate around the center of the image
            double x0 = (double) (c - imageWidth/2);
            double y0 = (double) (r - imageHeight/2);

            // Rotate using rotation matrix
            double x1 = x0*cos(radians) - y0*sin(radians);
            double y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (imageWidth/2);
            y1 += (double) (imageHeight/2);

            double rgb[3];
            BilinearInterpolation(buffer, x1, y1, rgb);

            // Note: "image[r*imageWidth+c] = rgb" merely copies the head pointers of the arrays, not the values
            image[r*imageWidth+c][0] = rgb[0];
            image[r*imageWidth+c][1] = rgb[1];
            image[r*imageWidth+c][2] = rgb[2];
        }
}

/**************************************************
 TASK 8
**************************************************/

// Find the peaks of the edge responses perpendicular to the edges
void MainWindow::FindPeaksImage(double** image, double thres)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
 * thres: threshold value for magnitude
*/
{
    // Add your code here
}

/**************************************************
 TASK 9 (a)
**************************************************/

// Perform K-means clustering on a color image using random seeds
void MainWindow::RandomSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
}

/**************************************************
 TASK 9 (b)
**************************************************/

// Perform K-means clustering on a color image using seeds from the image itself
void MainWindow::PixelSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
}


/**************************************************
 EXTRA CREDIT TASKS
**************************************************/

// Perform K-means clustering on a color image using the color histogram
void MainWindow::HistogramSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
}

// Apply the median filter on a noisy image to remove the noise
void MainWindow::MedianImage(double** image, int radius)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * radius: radius of the kernel
*/
{
    // Add your code here
}

// Apply Bilater filter on an image
void MainWindow::BilateralImage(double** image, double sigmaS, double sigmaI)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigmaS: standard deviation in the spatial domain
 * sigmaI: standard deviation in the intensity/range domain
*/
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

// Perform the Hough transform
void MainWindow::HoughImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}

// Perform smart K-means clustering
void MainWindow::SmartKMeans(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}
