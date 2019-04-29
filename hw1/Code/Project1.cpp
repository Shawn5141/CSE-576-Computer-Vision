#include <iostream>
#include <assert.h>
#include <cmath>
#include <vector>
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

    int buffer_imageWidth=imageWidth+kernelWidth-1;
    int buffer_imageHeight=imageHeight+kernelHeight-1;


    //Create a buffer with kernel size added
    int buffer_size=buffer_imageHeight*buffer_imageWidth;
    double **buffer = new double *[buffer_imageHeight*buffer_imageWidth] ;
    for (int M=0;M<buffer_size;M++){
        buffer[M]=new double [3];
    }




    // For each pixel
    for (int r = 0; r < buffer_imageHeight; r++){
        for (int c = 0; c < buffer_imageWidth; c++)
        {
            // Get a pixel from the QImage form of the
            if (r< int(kernelHeight/2) || r>=imageHeight+int(kernelHeight/2) || (c<int(kernelWidth/2) || c>=imageWidth+int(kernelWidth/2))){
                buffer[r*buffer_imageWidth+c][0]=0.0 ;
                buffer[r*buffer_imageWidth+c][1]=0.0 ;
                buffer[r*buffer_imageWidth+c][2]=0.0 ;


             }
            else{
                int _r= r-int(kernelHeight/2);
                int _c= c-int(kernelWidth/2);


                 // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively  
                 buffer[r*buffer_imageWidth+c][0]=Image[_r*imageWidth+_c][0] ;
                 buffer[r*buffer_imageWidth+c][1]=Image[_r*imageWidth+_c][1] ;
                 buffer[r*buffer_imageWidth+c][2]=Image[_r*imageWidth+_c][2] ;
            }


         }


}



    for (int r = 0; r < imageHeight; r++){
        for (int c = 0; c < imageWidth; c++){
            double value[3] = {0.0,0.0,0.0};

            for(int rd=-int(kernelHeight/2);rd<=int(kernelHeight/2);rd++){
                for(int cd=-int(kernelWidth/2);cd<=int(kernelWidth/2);cd++){

                    // Get the value of the kernel
                    double w = kernel[kernelWidth*(rd+int(kernelWidth/2))+(cd+int(kernelHeight/2))];
                    //std::cout<<kernel[kernelWidth*(rd+int(kernelWidth/2))+(cd+int(kernelHeight/2))]<<" ";
                    int index_r=r+int(kernelHeight/2)+rd;
                    int index_c=c+int(kernelWidth/2)+cd;
                    //std::cout<<"total "<<buffer[0][0]<<std::endl;

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
        }

    }




std::cout<<"Convolution fininished"<<std::endl;


for (int l=0;l<buffer_size;l++)
    delete [] buffer[l];
delete [] buffer;
delete [] kernel;



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

            int shift_x=x-int(kernelHeight/2);
            int shift_y=y-int(kernelWidth/2);


         kernel[x*kernelWidth+y] = (exp(-(shift_x*shift_x+shift_y*shift_y)/2/(sigma*sigma)))/2/M_PI/(sigma*sigma);

        }

    }
    //Normalized Kernel
    NormalizeKernel(kernel, kernelWidth, kernelHeight);



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
    for(int x=0;x<kernelHeight;x++){
        for(int y=0;y<kernelWidth;y++){

            int shift_x=x-int(kernelHeight/2);
            int shift_y=y-int(kernelWidth/2);
            //Horizontal vector with GAUSSAIN value
            if (y==int(kernelWidth/2)){
                kernel[x*kernelWidth+y] = (exp(-(shift_x*shift_x+shift_y*shift_y)/2/(sigma*sigma)))/2/M_PI/(sigma*sigma);
            }else{
                kernel[x*kernelWidth+y]=0.0;
            }
        }

    }

    //Normalized Kernel
    NormalizeKernel(kernel, kernelWidth, kernelHeight);


    Convolution(Image, kernel,  kernelWidth,  kernelHeight,  add);

    double *kernel1 =new double [kernelWidth*kernelHeight];
    for(int x=0;x<kernelHeight;x++){
        for(int y=0;y<kernelWidth;y++){

            int shift_x=x-int(kernelHeight/2);
            int shift_y=y-int(kernelWidth/2);
            //Vertical vector with GAUSSAIN value

            if (y==int(kernelWidth/2)){
                kernel1[x*kernelWidth+y] = (exp(-(shift_x*shift_x+shift_y*shift_y)/2/(sigma*sigma)))/2/M_PI/(sigma*sigma);
            }else{
                kernel1[x*kernelWidth+y]=0.0;
            }
        }

    }

    //Normalized Kernel
    NormalizeKernel(kernel1, kernelWidth, kernelHeight);

    Convolution(Image, kernel1,  kernelWidth,  kernelHeight,  add);



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
    // Kernel value for first derivative
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
    // Kernel value for first derivative
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
    // Kernel value for second derivative
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
    double **buffer=new double *[imageHeight*imageWidth];
    for (int i=0;i<imageHeight*imageWidth;i++)
        buffer[i]= new double[3];
    for (int r = 0; r < imageHeight; r++){
        for (int c = 0; c < imageWidth; c++){
            buffer[r*imageWidth+c][0]=Image[r*imageWidth+c][0];
            buffer[r*imageWidth+c][1]=Image[r*imageWidth+c][1];
            buffer[r*imageWidth+c][2]=Image[r*imageWidth+c][2];
        }
    }



    SecondDerivImage(Image,sigma);
    for (int r = 0; r < imageHeight; r++){
        for (int c = 0; c < imageWidth; c++){
            Image[r*imageWidth+c][0]=buffer[r*imageWidth+c][0]*alpha-Image[r*imageWidth+c][0]+128.0;
            Image[r*imageWidth+c][1]=buffer[r*imageWidth+c][1]*alpha-Image[r*imageWidth+c][1]+128.0;
            Image[r*imageWidth+c][2]=buffer[r*imageWidth+c][2]*alpha-Image[r*imageWidth+c][2]+128.0;
        }

    }
    for (int l=0;l<imageHeight*imageWidth;l++)
        delete [] buffer[l];
    delete [] buffer;

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
    double *kernel_x = new double[9];
    kernel_x[0]=-1.0;
    kernel_x[1]=0.0;
    kernel_x[2]=1.0;
    kernel_x[3]=-2.0;
    kernel_x[4]=0.0;
    kernel_x[5]=2.0;
    kernel_x[6]=-1.0;
    kernel_x[7]=0.0;
    kernel_x[8]=1.0;

    //Sobel kernel (horizontal Edge)
    double *kernel_y = new double[9];
    kernel_y[0]=1.0;
    kernel_y[1]=2.0;
    kernel_y[2]=1.0;
    kernel_y[3]=0.0;
    kernel_y[4]=0.0;
    kernel_y[5]=0.0;
    kernel_y[6]=-1.0;
    kernel_y[7]=-2.0;
    kernel_y[8]=-1.0;


    int kernelWidth =3;
    int kernelHeight=3;
     //NormalizeKernel(kernel_x, kernelWidth, kernelHeight);
     //NormalizeKernel(kernel_y, kernelWidth, kernelHeight);


    int buffer_imageWidth=imageWidth+kernelWidth-1;
    int buffer_imageHeight=imageHeight+kernelHeight-1;


    //Create a buffer with kernel size added
    int buffer_size=buffer_imageHeight*buffer_imageWidth;
    double **buffer1 = new double *[buffer_imageHeight*buffer_imageWidth] ;
    for (int M=0;M<buffer_size;M++){
        buffer1[M]=new double [3];
    }

    double **buffer2 = new double *[buffer_imageHeight*buffer_imageWidth] ;
    for (int M=0;M<buffer_size;M++){
        buffer2[M]=new double [3];
    }




    // For each pixel

    for (int r = 0; r < buffer_imageHeight; r++){
        for (int c = 0; c < buffer_imageWidth; c++)
        {

            if (r< int(kernelHeight/2) || r>=imageHeight+int(kernelHeight/2) || (c<int(kernelWidth/2) || c>=imageWidth+int(kernelWidth/2))){
                buffer1[r*buffer_imageWidth+c][0]=0.0 ;
                buffer2[r*buffer_imageWidth+c][0]=0.0 ;

                std::cout<<0;

             }
            else{
                int _r= r-int(kernelHeight/2);
                int _c= c-int(kernelWidth/2);
                // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively

                 buffer1[r*buffer_imageWidth+c][0]=Image[_r*imageWidth+_c][0] ;
                 buffer2[r*buffer_imageWidth+c][0]=Image[_r*imageWidth+_c][0] ;




            }


         }


}



    for (int r = 0; r < imageHeight; r++){
        for (int c = 0; c < imageWidth; c++){
            double value_x = 0.0;
            double value_y = 0.0;

            for(int rd=-int(kernelHeight/2);rd<=int(kernelHeight/2);rd++){
                for(int cd=-int(kernelWidth/2);cd<=int(kernelWidth/2);cd++){

                    // Get the value of the kernel
                    double w_x = kernel_x[kernelWidth*(rd+int(kernelWidth/2))+(cd+int(kernelHeight/2))];
                    double w_y = kernel_y[kernelWidth*(rd+int(kernelWidth/2))+(cd+int(kernelHeight/2))];
                    int index_r=r+int(kernelHeight/2)+rd;
                    int index_c=c+int(kernelWidth/2)+cd;
                    //add weight
                    value_x+=(buffer1[index_r*buffer_imageWidth+index_c][0]*w_x);
                    value_y+=(buffer2[index_r*buffer_imageWidth+index_c][0]*w_y);

                }

            }
            // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
            //Divide the both value by 8
            value_x/=8;
            value_y/=8;

            double mag = sqrt(value_x*value_x+value_y*value_y);
            double orien = atan2(value_y,value_x);
            Image[r*imageWidth+c][0] = mag*4.0*((sin(orien) + 1.0)/2.0);
            Image[r*imageWidth+c][1] = mag*4.0*((cos(orien) + 1.0)/2.0);
            Image[r*imageWidth+c][2] = mag*4.0 - Image[r*imageWidth+c][0] - Image[r*imageWidth+c][1];



        }

    }




for (int l=0;l<buffer_size;l++){
    delete [] buffer1[l];
    delete [] buffer2[l];
}

delete [] buffer1;
delete [] buffer2;
delete [] kernel_x;
delete [] kernel_y;




    // Use the following 3 lines of code to set the image pixel values after computing magnitude and orientation
    // Here 'mag' is the magnitude and 'orien' is the orientation angle in radians to be computed using atan2 function
    // (sin(orien) + 1)/2 converts the sine value to the range [0,1]. Similarly for cosine.


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
    int x1=int(x);
    int y1=int(y);
    int x2=x1+1;
    int y2=y1+1;

      for (int i=0;i<3;i++){


        if (x>0 && x<imageWidth-1 && y>0 && y<imageHeight-1 ){
              rgb[i]=(x2-x)*(y-y1)*image[y2*imageWidth+x1][i]+(x-x1)*(y-y1)*image[y2*imageWidth+x2][i]\
                    +(x2-x)*(y2-y)*image[y1*imageWidth+x1][i]+(x-x1)*(y2-y)*image[y1*imageWidth+x2][i];


         }
        else{
            rgb[i]=0.0;
        }


    }

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
            //std::cout<<"x1"<<x1<<" y1 "<<y1<<std::endl;
            x1 += (double) (imageWidth/2);
            y1 += (double) (imageHeight/2);

            double rgb[3];
            //std::cout<<"x1"<<x1<<" y1 "<<y1<<std::endl;
            BilinearInterpolation(buffer, x1, y1, rgb);

            // Note: "image[r*imageWidth+c] = rgb" merely copies the head pointers of the arrays, not the values
            image[r*imageWidth+c][0] = rgb[0];
            image[r*imageWidth+c][1] = rgb[1];
            image[r*imageWidth+c][2] = rgb[2];
        }



    for (int l=0;l<imageWidth*imageHeight;l++){
        delete [] buffer[l];

    }

    delete [] buffer;

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
    SobelImage(image);
    double** buffer=new double *[imageWidth*imageHeight];


    for(int i=0;i<imageWidth*imageHeight;i++){
        buffer[i]=new double [3];
    }
    double rgb1[3]={0,0,0};
    double rgb2[3]={0,0,0};

    for (int r=0 ;r<imageHeight;r++){
        for (int c=0;c<imageWidth;c++){
            buffer[r*imageWidth+c][0]=(image[r*imageWidth+c][0]+image[r*imageWidth+c][1]+image[r*imageWidth+c][2])/4.0;
            buffer[r*imageWidth+c][1]=(image[r*imageWidth+c][0]+image[r*imageWidth+c][1]+image[r*imageWidth+c][2])/4.0;
            buffer[r*imageWidth+c][2]=(image[r*imageWidth+c][0]+image[r*imageWidth+c][1]+image[r*imageWidth+c][2])/4.0;

            double mag=(image[r*imageWidth+c][0]+image[r*imageWidth+c][1]+image[r*imageWidth+c][2])/4.0;
            double sin_orien=image[r*imageWidth+c][0]/2/mag-1;
            double cos_orien=image[r*imageWidth+c][1]/2/mag-1;
            double e1x=c+cos_orien;
            double e1y=r+sin_orien;
            double e2x=c+cos_orien;
            double e2y=r-sin_orien;
            if (e1x>0 && e2x>0 && e1x<imageWidth && e2x<imageWidth &&\
                    e1y>0 && e2y>0 && e1y<imageHeight && e2y<imageHeight  ){
            BilinearInterpolation(buffer,e1x,e1y,rgb1);
            BilinearInterpolation(buffer,e2x,e2y,rgb2);
            }
            if (buffer[r*imageWidth+c][0]>thres and buffer[r*imageWidth+c][0]>rgb1[0] && buffer[r*imageWidth+c][0] >rgb2[0]){
                image[r*imageWidth+c][0] =255.0 ;
                image[r*imageWidth+c][1] =255.0 ;
                image[r*imageWidth+c][2] =255.0 ;

            }else{
                image[r*imageWidth+c][0] =0.0 ;
                image[r*imageWidth+c][1] =0.0 ;
                image[r*imageWidth+c][2] =0.0 ;

            }
        }


    }

    for (int l=0;l<imageHeight*imageWidth;l++)
        delete [] buffer[l];
    delete [] buffer;






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
    //"(rand() % 256, rand() % 256, rand() % 256)" as RGB values to create #num_clusters centers
    double **rgb=new double *[num_clusters];
    int epsilon=30;
    int max_iteration_num=100;
    //Create rgb buffer to save the seed value
    for (int i=0;i<num_clusters;i++){
        rgb[i]=new double[3];
        for(int j=0;j<3;j++)
            rgb[i][j]=rand()%256;
    }

    double old_sum=1000;
    int count=0;
    for (int ii=0;ii<max_iteration_num;ii++){

        //Total_distance equal to the sum of distance between red green blue
        double total_distance=0;

        //Create a buffer to save the pixel postion in each cluster
        std::vector<std::vector <int> > vec;
        for(int i = 0; i < num_clusters; i++)
        {
            std::vector<int> row_1;
            vec.push_back(row_1);
        }


        for (int r=0;r<imageHeight;r++){
            for (int c=0;c<imageWidth;c++){
                int min_index=0;
                int min_position=0;
                double min=1000.0;
                //Compare each pixel value to the cluster cnter's value
                for (int k=0;k<num_clusters;k++){
                    double red=fabs(rgb[k][0]-image[r*imageWidth+c][0]);
                    double green=fabs(rgb[k][1]-image[r*imageWidth+c][1]);
                    double blue=fabs(rgb[k][2]-image[r*imageWidth+c][2]);

                    total_distance=red+green+blue;
                    //Find the lowest distance in the for loop
                    if (red+green+blue<min){
                        min=red+green+blue;
                        min_index=k;
                        min_position=r*imageWidth+c;

                    }


                }
                //Assign the pixel postion to the cluster which has  lowest distance relative to the pixel
                vec[min_index].push_back(min_position);
                //std::cout<<std::endl;
        }
    }
        double _mean_r;
        double _mean_g;
        double _mean_b;
        int l=0;
        for (auto row:vec){
            double mean_r=0.0;
            double mean_g=0.0;
            double mean_b=0.0;



            //Avoid the empty cluster
            if (row.size()!=0){

                for (auto col:row){
                    mean_r+=image[col][0];
                    mean_g+=image[col][1];
                    mean_b+=image[col][2];


                 }
                //Find the mean of each coler
                _mean_r=mean_r/row.size();
                _mean_g=mean_g/row.size();
                _mean_b=mean_b/row.size();

                //Assign to new seed
                rgb[l][0]=_mean_r;
                rgb[l][1]=_mean_g;
                rgb[l][2]=_mean_b;


            }else{
                //If the cluster is empty, assign zero to the seed
                _mean_r=0;
                _mean_g=0;
                _mean_b=0;
                rgb[l][0]=_mean_r;
                rgb[l][1]=_mean_g;
                rgb[l][2]=_mean_b;



            }
            //Assign value to the image at each iteration
            for (auto col:row){

                image[col][0]=rgb[l][0];
                image[col][1]=rgb[l][1];
                image[col][2]=rgb[l][2];
                //std::cout<<col<<" ";


             }


            l+=1;


        }
        //Break the iteration if meet the condition
        if(0.001>fabs(old_sum-total_distance)){
            count+=1;
            if (count==20){
                break;
            }
        }
        old_sum=total_distance;

    }

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
    int r;
    int c;


    typedef std::vector<std::tuple<double,double,double>> rgb_type;
    rgb_type rgb;
    //rgb_type::const_iterator irgb = rgb.begin();
    r=rand()%imageHeight;
    c=rand()%imageWidth;


    double red=0;
    double green=0;
    double blue=0;

    do{
        //Create the seed from the image by randomly create r and c
        r=rand()%imageHeight;
        c=rand()%imageWidth;
        double tmp1=image[r*imageWidth+c][0];
        double tmp2=image[r*imageWidth+c][1];
        double tmp3=image[r*imageWidth+c][2];

    //Compare the distance between previous seed and new seed, if distance is different in 100, assign the value to the new seed
    if((fabs(tmp1-red)+fabs(tmp2-green)+fabs(tmp3-blue))>100){
       auto t = std::make_tuple(tmp1,tmp2,tmp3);
       rgb.push_back(t);

       }
    }while(rgb.size()<num_clusters);


    double old_sum=1000;
    int count=0;
    int max_iteration_num=100;
    for (int ii=0;ii<max_iteration_num;ii++){
        double total_distance=0;


        std::vector<std::vector <int> > vec;


        //Create new 2d cluster vector

        for(int i = 0; i < num_clusters; i++)
        {
            std::vector<int> row_1;
            vec.push_back(row_1);
        }

        for (int r=0;r<imageHeight;r++){
            for (int c=0;c<imageWidth;c++){
                int min_index=0;
                int min_position=0;
                double min=1000.0;
                for (int k=0;k<num_clusters;k++){
                    double red=fabs(std::get<0>(rgb[k])-image[r*imageWidth+c][0]);
                    double green=fabs(std::get<1>(rgb[k])-image[r*imageWidth+c][1]);
                    double blue=fabs(std::get<2>(rgb[k])-image[r*imageWidth+c][2]);


                    total_distance=red+green+blue;
                    //Find the lowest distance in the for loop
                    if (red+green+blue<min){
                        min=red+green+blue;
                        min_index=k;
                        min_position=r*imageWidth+c;

                    }


                }
                //Assign the pixel postion to the cluster which has  lowest distance relative to the pixel
                vec[min_index].push_back(min_position);

        }
    }
        double _mean_r;
        double _mean_g;
        double _mean_b;
        int l=0;
        for (auto row:vec){
            double mean_r=0.0;
            double mean_g=0.0;
            double mean_b=0.0;
            //Avoid the empty cluster
            if (row.size()!=0){
                //
                for (auto col:row){

                    mean_r+=image[col][0];
                    mean_g+=image[col][1];
                    mean_b+=image[col][2];

                 }
                //Find the mean of each coler
                _mean_r=mean_r/row.size();
                _mean_g=mean_g/row.size();
                _mean_b=mean_b/row.size();


                std::get<0>(rgb[l])=_mean_r;
                std::get<1>(rgb[l])=_mean_g;
                std::get<2>(rgb[l])=_mean_b;


            }else{
                //If the cluster is empty, assign zero to the seed
                _mean_r=0;
                _mean_g=0;
                _mean_b=0;
                std::get<0>(rgb[l])=_mean_r;
                std::get<1>(rgb[l])=_mean_g;
                std::get<2>(rgb[l])=_mean_b;

            }

            for (auto col:row){
                //Assign value to the image at each iteration
                image[col][0]=std::get<0>(rgb[l]);
                image[col][1]=std::get<1>(rgb[l]);
                image[col][2]=std::get<2>(rgb[l]);
             }

            //Increase increment to go through the cluster
            l+=1;


        }
        //Break the iteration if meet the condition
        if(0.001>fabs(old_sum-total_distance)){
            count+=1;
            if (count==20){
                //std::cout<<"break at :"<<epsilon*num_clusters<<"   "<<total_distance<<std::endl;
                break;
            }
        }
        old_sum=total_distance;
        //std::cout<<"total_distance :"<<total_distance<<std::endl;

    }


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
