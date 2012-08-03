#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <cmath>



using namespace cv;
using namespace std;

typedef pair<unsigned char, float> weightedNbr;
typedef vector<weightedNbr> weightedNbrs;
typedef float (*nbrfunc)(const weightedNbrs&, unsigned char);



float dist(Point2i I, Point2i J){
    return abs(I.x - J.x) + abs(I.y - J.y);
}


float mean(const weightedNbrs& nbrs){
    float mean = 0;
    float var = 0;
    for (int i = 0; i < nbrs.size(); i++){
        mean = mean + nbrs[i].first * nbrs[i].second;
    }
    return mean;
}

float variance(const weightedNbrs& nbrs, float avg){
    float var = 0;
    for (int i = 0; i < nbrs.size(); i++){
        var = var + nbrs[i].second * pow(nbrs[i].first - avg, 2);
    }
    return var;
}


float mvl(const weightedNbrs& wnbrs, unsigned char val){
   float avg = mean(wnbrs);
   float var = variance(wnbrs, avg);
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
   for (int i = 0; i < wnbrs.size() ; i++){
       total = 0;
       for (int j = 0; j < 256; j++){
           total = total + exp(beta * abs(j - wnbrs[i].first));
       }
       pred = pred + wnbrs[i].second * exp(beta * abs(val - wnbrs[i].first)) / total;
   }
   return pred;
}



Point2i boundarypt(Point2i ind, int r, int s){
    int d = s % (2 * r);
    if (s <= 2 * r){
        return Point2i(ind.x - r, ind.y - r + d);
    }
    else if ((2 * r < s) && (s <= 4 * r)){
        return Point2i(ind.x - r + d, ind.y + r);
    }
    else if ((4 * r < s) && (s <= 6 * r)){
        return Point2i(ind.x + r, ind.y + r - d);
    }
    else if (6 * r <= s){
        return Point2i(ind.x + r - d, ind.y - r);
    }
}


void nearestNeighbors(Point2i ind, Mat data, Mat_<unsigned char>& mask, weightedNbrs& wnbrs){
    int m = data.rows;
    int n = data.cols;
    int r = 1;
    int i, j;
    float wt;
    float wtsum = 0;
    Point2i bpt;
    weightedNbrs::size_type k = 0;
    vector<float> wts;
    while (k < wnbrs.size()){
        for (int s = 0; s <= 8 * r; s++){
            bpt = boundarypt(ind, r, s);
            i = bpt.x;
            j = bpt.y;
            if (k == wnbrs.size()){
                break;
            }
            else if ((0 <= i) && (i < m) && (0 <= j) && (j < n)){
                if (mask(i, j) == 1){
                    wnbrs[k].first = data.at<unsigned char>(i, j);
                    wt = 1. / dist(ind, bpt);
                    wtsum = wtsum + wt;
                    wts.push_back(wt);
                    k++;
                }
            }
        }
        r++;
    }
    for (int k = 0; k < wnbrs.size(); k++){
        wnbrs[k].second = wts[k] / wtsum;
    }

}



Mat process_randomly(Mat& data, nbrfunc func, int num_nbrs = 4){
    int m = data.rows;
    int n = data.cols;
    Mat prediction(m, n, CV_32FC1);
    Mat_<unsigned char> seen_mask(m, n, (unsigned char) 0);
    vector<Point2i> indices;
    weightedNbrs wnbrs(num_nbrs);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            indices.push_back(Point2i(i, j));
        }
    }
    random_shuffle(indices.begin(), indices.end());
    int count = 0;
    for (int k = 0; k < indices.size(); k++){
        if (k < num_nbrs){
            prediction.at<float>(indices[k]) = 1. / 256;
        }
        else{
            nearestNeighbors(indices[k], data, seen_mask, wnbrs);
            prediction.at<float>(indices[k]) = func(wnbrs, data.at<unsigned char>(indices[k]));
        }
        seen_mask(indices[k]) = 1;
        count ++;
        if (count % 1000 == 0){
            cout << count << '\n';
        }
    }
    return prediction;
}



float approx_cost(const Mat& prediction){
    float cost = 0;
    float prob;
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
    Mat M = imread("../test-images/test-images-513/001-0.png", 0);
    Mat prediction = process_randomly(M, mvl);
    cout << approx_cost(prediction) << '\n' << endl;
    return 0;
}
