#include <iostream>

#include "MyVideoCapture.h"

void fft(cv::Mat &src){

    int cx = src.cols / 2;
    int cy = src.rows / 2;

    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

}

int main() {

    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        std::cout << "Cannot open camera.";
        return -1;
    }

    while(true){
        cv::Mat frame;
        cap.read(frame);
        cv::cvtColor(frame, frame, CV_RGB2GRAY);
        //cv::imshow("cam", frame);

        cv::Mat padded;                            //expand input image to optimal size
        int m = cv::getOptimalDFTSize( frame.rows );
        int n = cv::getOptimalDFTSize( frame.cols ); // on the border add zero values
        cv::copyMakeBorder(frame, padded, 0, m - frame.rows, 0, n - frame.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

        cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
        cv::Mat magI = planes[0];

        magI += cv::Scalar::all(1);                    // switch to logarithmic scale
        cv::log(magI, magI);

        // crop the spectrum, if it has an odd number of rows or columns
        magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = magI.cols/2;
        int cy = magI.rows/2;

        cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

        cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);

        cv::normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
        // viewable image form (float between values 0 and 1).

        cv::imshow("Input Image"       , frame);    // Show the result
        cv::imshow("spectrum magnitude", magI);

        if(cv::waitKey(30) >= 0){
            break;
        }

    }

    return 0;
}