import numpy as np

Matrix = [[i for i in range(9)] for j in range(3)]
List= [" " for i in range(9)]


kernelHeight=3
kernelWidth=3
_buffer=[[0.0 for i in range(len(Matrix[0])+kernelWidth-1)]for j in range(len(Matrix)+kernelHeight-1)]
buffer_imageWidth=3+kernelWidth-1
buffer_imageHeight=3+kernelHeight-1
imageHeight=3
imageWidth=3

for i in range(3):
    for j in range(3):
        List[i*3+j]=Matrix[i][j]


Matrix=[1 for j in range(imageHeight*imageWidth)]
_buffer = [0 for j in range(buffer_imageHeight*buffer_imageWidth)] 

#print(_buffer)
for r in range( buffer_imageHeight):
        for c in range(buffer_imageWidth):
            #// Get a pixel from the QImage form of the

            if (r< int(imageHeight/2) or r>=imageHeight+int(kernelHeight/2) or (c<int(imageWidth/2) or c>=imageWidth+int(kernelWidth/2))):
                print("outside",r,c)
                _buffer[r*buffer_imageWidth+c]=0.0 
                #_buffer[r*buffer_imageWidth+c][1]=0.0 
                #_buffer[r*buffer_imageWidth+c][2]=0.0 
             
            else:
                _r= r-int(kernelHeight/2)
                _c= c-int(kernelWidth/2)

                 # Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
                 #//std::cout<<r*buffer_imageWidth+c<<std::endl;
               
                _buffer[r*buffer_imageWidth+c]=Matrix[_r*imageWidth+_c] 
               
                print("insider",r,c)
 #_buffer[r*buffer_imageWidth+c][1]=Matrix[_r*imageWidth+_c][1] ;
                #_buffer[r*buffer_imageWidth+c][2]=Matrix[_r*imageWidth+_c][2] ;
            
            #//std::cout<<" buffer :"<<buffer[m][n]<<std::endl;
         





'''
for  m in range(buffer_imageHeight):
    for n in range(buffer_imageWidth):
	   if m< int(imageHeight)/2 or m>=imageHeight+int(kernelHeight/2) or n< int(imageWidth/2) or n>=imageWidth+int(kernelWidth/2):

                   _buffer[m][n]=0.0
                   print()
                   print("H",imageHeight/2,imageHeight+int(kernelHeight/2))
                   print("W",imageWidth/2,imageWidth+int(kernelWidth/2))
                   print("outside",m,n)
            
           else:
               
               print("H",imageHeight/2,imageHeight+int(kernelHeight/2))
               
               print("W",imageWidth/2,imageWidth+int(kernelWidth/2))
             
               print("insider",m,n)   
	       _buffer[m][n]=Matrix[m-int(kernelHeight/2)][n-int(kernelWidth/2)]
               #print("insider",m,n)
           
'''           

for r in range(imageHeight):
    for c in range(imageWidth):
        value = 0.0
        for rd in range(-int(kernelHeight/2),int(kernelHeight/2)+1):
            for cd in range(-int(kernelWidth/2),int(kernelWidth/2+1)):

                # Get the value of the kernel
                #weight = kernel[kernelWidth*(rd+int(kernelHeight/2))+(cd+int(kernelWidth/2))];
                #std::cout<<"c "<<c+int(kernelWidth/2)+cd<<std::endl;
                #std::cout<<"r "<<r+int(kernelHeight/2)+rd<<std::endl;
     
                index_r=r+int(kernelHeight/2)+rd;
                index_c=c+int(kernelWidth/2)+cd;
                print("index_r and c",index_r*buffer_imageWidth+index_c)
                #print(_buffer)
                value+=_buffer[index_r*buffer_imageWidth+index_c]
                #value[1]+=_buffer[index_r*buffer_imageWidth+index_c][1];
                #value[2]+=_buffer[index_r*buffer_imageWidth+index_c][2];


                #print("r ",r+int(index_r))
                #print("c ",c+int(index_c))
        print("value ",value)
        Matrix[r*imageWidth+c]=value

                
            #std::cout<<"value"<<value<<std::endl;
  





       
  
print(np.array(Matrix).reshape(1,3,-1))
print(np.array(_buffer).reshape(1,5,-1))

