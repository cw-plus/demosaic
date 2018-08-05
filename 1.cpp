/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"

#include <cstdio>
#include <vector>
#include <iostream>
#include <functional>

namespace cv
{

struct greaterThanPtr :
        public std::binary_function<const float *, const float *, bool>
{
    bool operator () (const float * a, const float * b) const
    { return *a > *b; }
};

#ifdef HAVE_OPENCL

struct Corner
{
    float val;
    short y;
    short x;

    bool operator < (const Corner & c) const
    {  return val > c.val; }
};

static bool ocl_goodFeaturesToTrack( InputArray _image, OutputArray _corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     InputArray _mask, int blockSize,
                                     bool useHarrisDetector, double harrisK )
{
    UMat eig, maxEigenValue;
    if( useHarrisDetector )
        cornerHarris( _image, eig, blockSize, 3, harrisK );
    else
        cornerMinEigenVal( _image, eig, blockSize, 3 );

    Size imgsize = _image.size();
    size_t total, i, j, ncorners = 0, possibleCornersCount =
            std::max(1024, static_cast<int>(imgsize.area() * 0.1));
    bool haveMask = !_mask.empty();
    UMat corners_buffer(1, (int)possibleCornersCount + 1, CV_32FC2);
    CV_Assert(sizeof(Corner) == corners_buffer.elemSize());
    Mat tmpCorners;

    // find threshold
    {
        CV_Assert(eig.type() == CV_32FC1);
        int dbsize = ocl::Device::getDefault().maxComputeUnits();
        size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

        int wgs2_aligned = 1;
        while (wgs2_aligned < (int)wgs)
            wgs2_aligned <<= 1;
        wgs2_aligned >>= 1;

        ocl::Kernel k("maxEigenVal", ocl::imgproc::gftt_oclsrc,
                      format("-D OP_MAX_EIGEN_VAL -D WGS=%d -D groupnum=%d -D WGS2_ALIGNED=%d%s",
                             (int)wgs, dbsize, wgs2_aligned, haveMask ? " -D HAVE_MASK" : ""));
        if (k.empty())
            return false;

        UMat mask = _mask.getUMat();
        maxEigenValue.create(1, dbsize, CV_32FC1);

        ocl::KernelArg eigarg = ocl::KernelArg::ReadOnlyNoSize(eig),
                dbarg = ocl::KernelArg::PtrWriteOnly(maxEigenValue),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
                cornersarg = ocl::KernelArg::PtrWriteOnly(corners_buffer);

        if (haveMask)
            k.args(eigarg, eig.cols, (int)eig.total(), dbarg, maskarg);
        else
            k.args(eigarg, eig.cols, (int)eig.total(), dbarg);

        size_t globalsize = dbsize * wgs;
        if (!k.run(1, &globalsize, &wgs, false))
            return false;

        ocl::Kernel k2("maxEigenValTask", ocl::imgproc::gftt_oclsrc,
                       format("-D OP_MAX_EIGEN_VAL -D WGS=%d -D WGS2_ALIGNED=%d -D groupnum=%d",
                              wgs, wgs2_aligned, dbsize));
        if (k2.empty())
            return false;

        k2.args(dbarg, (float)qualityLevel, cornersarg);

        if (!k2.runTask(false))
            return false;
    }

    // collect list of pointers to features - put them into temporary image
    {
        ocl::Kernel k("findCorners", ocl::imgproc::gftt_oclsrc,
                      format("-D OP_FIND_CORNERS%s", haveMask ? " -D HAVE_MASK" : ""));
        if (k.empty())
            return false;

        ocl::KernelArg eigarg = ocl::KernelArg::ReadOnlyNoSize(eig),
                cornersarg = ocl::KernelArg::PtrWriteOnly(corners_buffer),
                thresholdarg = ocl::KernelArg::PtrReadOnly(maxEigenValue);

        if (!haveMask)
            k.args(eigarg, cornersarg, eig.rows - 2, eig.cols - 2, thresholdarg,
                  (int)possibleCornersCount);
        else
        {
            UMat mask = _mask.getUMat();
            k.args(eigarg, ocl::KernelArg::ReadOnlyNoSize(mask),
                   cornersarg, eig.rows - 2, eig.cols - 2,
                   thresholdarg, (int)possibleCornersCount);
        }

        size_t globalsize[2] = { eig.cols - 2, eig.rows - 2 };
        if (!k.run(2, globalsize, NULL, false))
            return false;

        tmpCorners = corners_buffer.getMat(ACCESS_RW);
        total = std::min<size_t>(tmpCorners.at<Vec2i>(0, 0)[0], possibleCornersCount);
        if (total == 0)
        {
            _corners.release();
            return true;
        }
    }

    Corner* corner_ptr = tmpCorners.ptr<Corner>() + 1;
    std::sort(corner_ptr, corner_ptr + total);

    std::vector<Point2f> corners;
    corners.reserve(total);

    if (minDistance >= 1)
    {
         // Partition the image into larger grids
        int w = imgsize.width, h = imgsize.height;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);
        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            const Corner & c = corner_ptr[i];
            bool good = true;

            int x_cell = c.x / cell_size;
            int y_cell = c.y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for( int yy = y1; yy <= y2; yy++ )
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector<Point2f> &m = grid[yy * grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = c.x - m[j].x;
                            float dy = c.y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)c.x, (float)c.y));

                corners.push_back(Point2f((float)c.x, (float)c.y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            const Corner & c = corner_ptr[i];

            corners.push_back(Point2f((float)c.x, (float)c.y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners ) 
                break;
        }
    }

    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
    return true;
}

#endif

}

void cv::goodFeaturesToTrack( InputArray _image, OutputArray _corners,
                              int maxCorners, double qualityLevel, double minDistance,
                              InputArray _mask, int blockSize,
                              bool useHarrisDetector, double harrisK )
{
    CV_Assert( qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0 );
    CV_Assert( _mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)) );

	//opencl 不用看
    CV_OCL_RUN(_image.dims() <= 2 && _image.isUMat(),
               ocl_goodFeaturesToTrack(_image, _corners, maxCorners, qualityLevel, minDistance,
                                    _mask, blockSize, useHarrisDetector, harrisK))
    //如果需要对_image全图操作，则给_mask传入cv::Mat()，否则传入感兴趣区域
    Mat image = _image.getMat(), eig, tmp;  //eig存储每个像素协方差矩阵的最小特征值，tmp用来保存经膨胀后的eig
    if (image.empty())  //异常处理
    {
        _corners.release();
        return;
    }

	// 计算特征值
    if( useHarrisDetector )
        cornerHarris( image, eig, blockSize, 3, harrisK );  //blockSize是计算2*2协方差矩阵的窗口大小，
	                                                        //sobel算子窗口为3，harrisK是计算Harris角点时需要的值
    else
        cornerMinEigenVal( image, eig, blockSize, 3 );  //https://blog.csdn.net/jaych/article/details/51146923
	                                                    //计算每个像素对应的协方差矩阵的最小特征值，保存在eig中


	// 特征值处理，去除不符合门限的特征点及局部最优点
    double maxVal = 0; //maxVal保存了eig的最大值
    minMaxLoc( eig, 0, &maxVal, 0, 0, _mask ); //minMaxLoc 查找全局最小和最大数组元素并返回它们的值和它们的位置。
/*寻找最值：minMaxLoc()函数
功能：查找全局最小和最大数组元素并返回它们的值和它们的位置。
void minMaxLoc(InputArray src, CV_OUT double* minVal,
                           CV_OUT double* maxVal=0, CV_OUT Point* minLoc=0,
                           CV_OUT Point* maxLoc=0, InputArray mask=noArray());
参数解释
参数1：InputArray类型的src，输入单通道数组（图像）。
参数2：double*类型的minVal，返回最小值的指针。若无须返回，此值置为NULL。
参数3：double*类型的maxVal，返回最大值的指针。若无须返回，此值置为NULL。
参数4：Point*类型的minLoc，返回最小位置的指针（二维情况下）。若无须返回，此值置为NULL。
参数5：Point*类型的maxLoc，返回最大位置的指针（二维情况下）。若无须返回，此值置为NULL。
参数6：InputArray类型的mask，用于选择子阵列的可选掩膜。
*/
    threshold( eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO );  //大于阈值保留原值，否则置0 。在这里参数0用不到
    
    dilate( eig, tmp, Mat());  //膨胀算法使图像扩大一圈。 膨胀：给图像中的对象边界添加像素
	//膨胀算法：用3X3的结构元素，扫描二值图像的每一个像素，用结构元素与其覆盖的二值图像做“与”运算，
	//如果都为0，结构图像的该像素为0，否则为1.结果：使二值图像扩大一圈。

    Size imgsize = image.size();
	//存放粗选出的角点地址
    std::vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    Mat mask = _mask.getMat();
    for( int y = 1; y < imgsize.height - 1; y++ )//因为膨胀了一圈所以可以从1-imgsize.height - 2都可以
    {
        const float* eig_data = (const float*)eig.ptr(y);//获得eig第y行的首地址
        const float* tmp_data = (const float*)tmp.ptr(y);//获得tmp第y行的首地址
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;  //如果使用mask则指向mask的第y行

        for( int x = 1; x < imgsize.width - 1; x++ )//列地址
        {
            float val = eig_data[x]; //指向特征矩阵的y行x列
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )//val == tmp_data[x]说明这是局部极大值
                tmpCorners.push_back(eig_data + x);//保存其位置
        }
    }
//-----------此分割线以上是根据特征值粗选出的角点，我们称之为弱角点----------//
//-----------此分割线以下还要根据minDistance进一步筛选角点，仍然能存活下来的我们称之为强角点----------//
	
    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr() );
//按特征值降序排列，注意这一步很重要，后面的很多编程思路都是建立在这个降序排列的基础上

    std::vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

	 //下面的程序有点稍微难理解，需要自己仔细想想

	//根据特征点之间的距离限制，筛选特征点
    if (minDistance >= 1) //minDistance：对于初选出的角点而言，如果在其周围minDistance范围内存在其他更强角点，则将此角点删除
    {
         // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);//向最近的整数取整 对一个double型的数进行四舍五入，并返回一个整型数
		
		//根据cell_size构建了一个矩形窗口grid(虽然下面的grid定义的是vector<vector>，
		//而并不是我们这里说的矩形窗口，但为了便于理解,还是将grid想象成一个grid_width * grid_height的矩形窗口比较好)，
		//除以cell_size说明grid窗口里相差一个像素相当于_image里相差minDistance个像素，至于为什么加上cell_size - 1后面会讲
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

		//二维数组grid用来保存获得的强角点坐标
        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;//平方，方面后面计算，省的开根号

		//那就变成归一化的minDistance要<=1????
        for( i = 0; i < total; i++ )// 刚刚粗选的弱角点，都要到这里来接收新一轮的考验
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr()); //tmpCorners中保存了角点的地址，eig.data返回eig内存块的首地址
            int y = (int)(ofs / eig.step);//角点在原图像中的行
            int x = (int)((ofs - y*eig.step)/sizeof(float));//在原图像中的列

            bool good = true;//先认为当前角点能接收考验，即能被保留下来

			//(x,y)是当前角点在img上的坐标，因为tmpCorners是存的相对起始位置的距离
            int x_cell = x / cell_size;//x_cell，y_cell是角点（y,x）在grid中的对应坐标
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;// (y_cell，x_cell）的4邻域像素
            int y1 = y_cell - 1;//现在知道为什么前面grid_width定义时要加上cell_size - 1了吧，这是为了使得（y,x）在grid中的4邻域像素都存在，也就是说(y_cell，x_cell）不会成为边界像素

            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

			// boundary check，再次确认x1,y1,x2或y2不会超出grid边界
            x1 = std::max(0, x1);//比较0和x1的大小
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

			 //记住grid中相差一个像素，相当于_image中相差了minDistance个像素
            for( int yy = y1; yy <= y2; yy++ ) //行     3*3的框
                for( int xx = x1; xx <= x2; xx++ ) //列
                {
                    std::vector <Point2f> &m = grid[yy*grid_width + xx]; //引用
                    //？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
                    if( m.size() )//如果(y_cell，x_cell)的4邻域像素，也就是(y,x)的minDistance邻域像素中已有被保留的强角点
                    {    //？？？？？？？？？？？？？？？？？           
                        for(j = 0; j < m.size(); j++) //当前角点周围的强角点都拉出来跟当前角点比一比
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;
//注意如果(y,x)的minDistance邻域像素中已有被保留的强角点，则说明该强角点是在(y,x)之前就被测试过的，
//又因为tmpCorners中已按照特征值降序排列（特征值越大说明角点越好），这说明先测试的一定是更好的角点，
//也就是已保存的强角点一定好于当前角点，所以这里只要比较距离，如果距离满足条件，可以立马扔掉当前测试的角点
                      
                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                corners.push_back(Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else//除了像素本身，没有哪个邻域像素能与当前像素满足minDistance < 1,因此直接保存粗选的角点
    {  //minDistance：对于初选出的角点而言，如果在其周围minDistance范围内存在其他更强角点，则将此角点删除
        for( i = 0; i < total; i++ ) //total = tmpCorners.size()
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr()); //求得偏置 用以划分行
            int y = (int)(ofs / eig.step); //粗选的角点在原图像中的行
			//step：每一行中所有元素的字节总量，单位字节；uchar是一个字节所以可以用来求解坐标
            int x = (int)((ofs - y*eig.step)/sizeof(float));//在图像中的列

            corners.push_back(Point2f((float)x, (float)y));//把角点坐标放到corners里
            ++ncorners;//计数加1
            if( maxCorners > 0 && (int)ncorners == maxCorners )//按照特征值从大到小取角点，直到达到最大角点数目
                break;
        }
    }
    //输出corners
    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

CV_IMPL void
cvGoodFeaturesToTrack( const void* _image, void*, void*,
                       CvPoint2D32f* _corners, int *_corner_count,
                       double quality_level, double min_distance,
                       const void* _maskImage, int block_size,
                       int use_harris, double harris_k )
{
    cv::Mat image = cv::cvarrToMat(_image), mask;
    std::vector<cv::Point2f> corners;

    if( _maskImage )
        mask = cv::cvarrToMat(_maskImage);

    CV_Assert( _corners && _corner_count );
    cv::goodFeaturesToTrack( image, corners, *_corner_count, quality_level,
        min_distance, mask, block_size, use_harris != 0, harris_k );

    size_t i, ncorners = corners.size();
    for( i = 0; i < ncorners; i++ )
        _corners[i] = corners[i];
    *_corner_count = (int)ncorners;
}

/* End of file. */
