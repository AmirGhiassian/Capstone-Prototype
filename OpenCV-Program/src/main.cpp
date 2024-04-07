#include <opencv2/opencv.hpp>

using namespace cv;
int main()
{
    Mat frame;

    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    while (1)
    {
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "No captured frame" << std::endl;
            break;
        }

        imshow("Frame", frame);

        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    imshow("Frame", frame);
    waitKey(0);

    return 0;
}