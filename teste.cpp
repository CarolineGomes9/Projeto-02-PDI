#include <interop/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("teste.png");
    if (img.empty()) {
        cerr << "Erro ao carregar a imagem!" << endl;
        return -1;
    }
    cout << "Imagem carregada com sucesso!" << endl;
    imshow("Imagem", img);
    waitKey(0);
    return 0;
}
