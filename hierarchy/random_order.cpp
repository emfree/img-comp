#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <cmath>


using namespace cv;
using namespace std;


float dist(Point2i I, Point2i J){
    return abs(I.x - J.x) + abs(I.y - J.y);
}




void boundarypt(const Point2i& index, Point2i& bpt, int r, int s){
    int d = s % (2 * r);
    if (s <= 2 * r){
        bpt.x = index.x - r;
        bpt.y = index.y - r + d;
    }
    else if ((2 * r < s) && (s <= 4 * r)){
        bpt.x = index.x - r + d;
        bpt.y = index.y + r;
    }
    else if ((4 * r < s) && (s <= 6 * r)){
        bpt.x = index.x + r;
        bpt.y = index.y + r - d;
    }
    else if (6 * r <= s){
        bpt.x = index.x + r - d;
        bpt.y = index.y - r;
    }
}



class WeightedNbrs {
    public:
    WeightedNbrs(int);
    void update(Point2i& ind, const Mat& data, Mat& mask);
    int num_nbrs;
    int *nbrs;
    float *weights;
    float mean();
    float variance();
    float mvl(unsigned char);
    private:
    float avg;
    float var;
};


WeightedNbrs::WeightedNbrs(int number_of_neighbors){
    nbrs = new int[number_of_neighbors];
    weights = new float[number_of_neighbors];
    num_nbrs = number_of_neighbors;
}

void WeightedNbrs::update(Point2i& ind, const Mat& data, Mat& mask){
    int m = data.rows;
    int n = data.cols;
    int r = 1;
    int i, j;
    float wt;
    float wtsum = 0;
    Point2i bpt;
    int k = 0;
    while (k < num_nbrs){
        for (int s = 0; s <= 8 * r; s++){
            boundarypt(ind, bpt, r, s);
            i = bpt.x;
            j = bpt.y;
            if (k == num_nbrs){
                break;
            }
            else if ((0 <= i) && (i < m) && (0 <= j) && (j < n)){
                if (mask.at<unsigned char>(i, j) > 0){
                    nbrs[k] = data.at<unsigned char>(i, j);
                    wt = 1. / dist(ind, bpt);
                    wtsum = wtsum + wt;
                    weights[k] = wt;
                    k++;
                }
            }
        }
        r++;
    }
    for (int k = 0; k < num_nbrs; k++){
        weights[k] = weights[k] / wtsum;
    }
}

float WeightedNbrs::mean(){
    avg = 0;
    for (int i = 0; i < num_nbrs; i++){
        avg = avg + nbrs[i] * weights[i];
    }
    return avg;
}

float WeightedNbrs::variance(){
    var = 0;
    for (int i = 0; i < num_nbrs; i++){
        var = var + weights[i] * pow(nbrs[i] - avg, 2);
    }
    return var;
}

float WeightedNbrs::mvl(unsigned char val){
    mean();
    variance();
    float beta, total;
    if (var > 0){
        beta = - 1. / sqrt(var / 2);
    }
    else{
       if (val == avg){
           return .5;
       }
       else{
           return .5 / 255;
       }
   }
   float pred = 0;
   for (int i = 0; i < num_nbrs ; i++){
       total = 0;
       for (int j = 0; j < 256; j++){
           total = total + exp(beta * abs(j - nbrs[i]));
       }
       pred = pred + weights[i] * exp(beta * abs( val - nbrs[i])) / total;
   }
   return pred;
}







Mat process_randomly(Mat& data, int num_nbrs = 4){
    namedWindow("test");
    int m = data.rows;
    int n = data.cols;
    Mat prediction(m, n, CV_32FC1);
    Mat seen_mask = Mat::zeros(m, n, CV_8UC1);
    vector<Point2i> indices;
    WeightedNbrs wnbrs(num_nbrs);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            indices.push_back(Point2i(i, j));
        }
    }
    random_shuffle(indices.begin(), indices.end());
    int count = 0;
    float pred;
    for (int k = 0; k < indices.size(); k++){
        if (k < num_nbrs){
            prediction.at<float>(indices[k]) = 1. / 256;
        }
        else{
            wnbrs.update(indices[k], data, seen_mask);
            pred = wnbrs.mvl(data.at<unsigned char>(indices[k]));
            prediction.at<float>(indices[k]) = pred;
        }
        seen_mask.at<unsigned char>(indices[k]) = 255;
        count ++;
        if (count % 1000 == 0){
            imshow("test", seen_mask);
            cout << count << '\n';
            waitKey(5);
        }
    }
    return prediction;
}



float approx_cost(const Mat& prediction){
    float cost = 0;
    float prob = 0;
    for (int i = 0; i < prediction.rows; i++){
        for (int j = 0; j < prediction.cols; j++){
            prob = prediction.at<float>(i, j);
            if (prob > 0){
                cost = cost - log2(prob);
            }
        }
    }
    return cost;
}






int main(){
    Mat M = imread("../test-images/test-images-513/002-0.png", 0);
    Mat prediction = process_randomly(M);
    cout << approx_cost(prediction) << '\n';
    return 0;
}
