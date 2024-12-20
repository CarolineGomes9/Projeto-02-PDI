#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Função para calcular histograma de uma imagem
Mat calcHistogram(const Mat &img) {
    Mat hist;
    int histSize = 256; 
    float range[] = {0, 256};
    const float *histRange = {range};

    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    return hist;
}

Mat drawHistogramStyled(const Mat &hist, const string &title, int width = 512, int height = 400) {
    Mat histImage(height + 50, width + 50, CV_8UC3, Scalar(255, 255, 255));
    normalize(hist, hist, 0, height, NORM_MINMAX);

    int bin_w = cvRound((double)width / hist.rows);

    for (int i = 1; i < hist.rows; i++) {
        line(histImage,
             Point(bin_w * (i - 1) + 25, height - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * i + 25, height - cvRound(hist.at<float>(i))),
             Scalar(0, 0, 255), 1, LINE_AA);
    }

    line(histImage, Point(25, height), Point(width + 25, height), Scalar(0, 0, 0), 1, LINE_AA); 
    line(histImage, Point(25, 0), Point(25, height), Scalar(0, 0, 0), 1, LINE_AA);

    for (int i = 0; i <= 256; i += 64) {
        putText(histImage, to_string(i), Point(bin_w * i / 256 * width + 20, height + 20),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    }
    for (int i = 0; i <= height; i += 100) {
        putText(histImage, to_string(height - i), Point(5, i + 5), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    }

    putText(histImage, title, Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

    return histImage;
}

// Aplica transformação logarítmica: s = c*log(1 + r)
Mat logTransform(const Mat &img, double c) {
    Mat imgFloat, imgLog;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    log(1 + imgFloat, imgLog);
    imgLog *= c;
    normalize(imgLog, imgLog, 0, 1, NORM_MINMAX);
    imgLog.convertTo(imgLog, CV_8U, 255.0);
    return imgLog;
}

// Aplica transformação de potência (Gamma): s = c*(r^gamma)
Mat gammaTransform(const Mat &img, double c, double gamma) {
    Mat imgFloat, imgGamma;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    pow(imgFloat, gamma, imgGamma);
    imgGamma *= c;
    normalize(imgGamma, imgGamma, 0, 1, NORM_MINMAX);
    imgGamma.convertTo(imgGamma, CV_8U, 255.0);
    return imgGamma;
}

Mat globalHistogramEqualization(const Mat &img) {
    Mat eq;
    equalizeHist(img, eq);
    return eq;
}

Mat localHistogramEqualization(const Mat &img, double clipLimit = 2.0, Size tileGridSize = Size(3, 3)) {
    Ptr<CLAHE> clahe = createCLAHE(clipLimit, tileGridSize);
    Mat claheImg;
    clahe->apply(img, claheImg);
    return claheImg;
}

// Função para calcular a função de transformação da equalização (CDF normalizada)
vector<int> calculateTransformFunction(const Mat &img) {
    Mat hist = calcHistogram(img);
    hist /= (img.rows * img.cols); 

    vector<float> cdf(256, 0.0f);
    cdf[0] = hist.at<float>(0);
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist.at<float>(i);
    }

    // Mapeia para [0, 255]
    vector<int> transform(256, 0);
    for (int i = 0; i < 256; i++) {
        transform[i] = cvRound(cdf[i] * 255);
    }
    return transform;
}

// Função para desenhar a função de transformação
// Eixo X: intensidade original [0..255]
// Eixo Y: intensidade equalizada [0..255]
Mat drawTransformFunction(const vector<int> &transform, const string &title, int width = 512, int height = 400) {
    Mat image(height + 50, width + 50, CV_8UC3, Scalar(255, 255, 255));

    int bin_w = cvRound((double)width / 256.0);

    for (int i = 1; i < 256; i++) {
        Point p1((i - 1)*bin_w + 25, height - transform[i-1]);
        Point p2(i*bin_w + 25, height - transform[i]);
        line(image, p1, p2, Scalar(0, 0, 255), 2, LINE_AA);
    }

    line(image, Point(25, height), Point(width + 25, height), Scalar(0, 0, 0), 1, LINE_AA);
    line(image, Point(25, 0), Point(25, height), Scalar(0, 0, 0), 1, LINE_AA);

    for (int i = 0; i <= 255; i += 64) {
        putText(image, to_string(i), Point(i*bin_w + 20, height + 20),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,0),1);
    }
    for (int i = 0; i <= 255; i += 64) {
        putText(image, to_string(255 - i), Point(5, i), 
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,0),1);
    }

    putText(image, title, Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,0), 2);
    return image;
}

void processImage(const string &inputPath, const string &outputDir, bool computeLocalEq=false) {
    Mat img = imread(inputPath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Erro ao abrir a imagem: " << inputPath << endl;
        return;
    }

    string baseName = fs::path(inputPath).stem().string();
    string histDir = outputDir + "/Histogramas";
    string imgDir = outputDir + "/Imagens_transformadas";
    string transfDir = outputDir + "/Funcoes_Transformacao";
    fs::create_directories(histDir);
    fs::create_directories(imgDir);
    fs::create_directories(transfDir);

    Mat logImg = logTransform(img, 1.0);
    Mat gammaImg = gammaTransform(img, 1.0, 0.4);
    Mat eqImg = globalHistogramEqualization(img);

    Mat histOriginal = calcHistogram(img);
    Mat histLog = calcHistogram(logImg);
    Mat histGamma = calcHistogram(gammaImg);
    Mat histEq = calcHistogram(eqImg);

    Mat histPlotOriginal = drawHistogramStyled(histOriginal, "Original");
    Mat histPlotLog = drawHistogramStyled(histLog, "Log Transformation");
    Mat histPlotGamma = drawHistogramStyled(histGamma, "Gamma Transformation");
    Mat histPlotEq = drawHistogramStyled(histEq, "Global Equalization");

    imwrite(imgDir + "/" + baseName + "_original.png", img);
    imwrite(imgDir + "/" + baseName + "_log_transform.png", logImg);
    imwrite(imgDir + "/" + baseName + "_gamma_transform.png", gammaImg);
    imwrite(imgDir + "/" + baseName + "_global_equalization.png", eqImg);

    imwrite(histDir + "/" + baseName + "_hist_original.png", histPlotOriginal);
    imwrite(histDir + "/" + baseName + "_hist_log.png", histPlotLog);
    imwrite(histDir + "/" + baseName + "_hist_gamma.png", histPlotGamma);
    imwrite(histDir + "/" + baseName + "_hist_eq.png", histPlotEq);

    vector<int> transformFunction = calculateTransformFunction(img);
    Mat transformPlot = drawTransformFunction(transformFunction, "Funcao de Transformacao Equalizacao");
    imwrite(transfDir + "/" + baseName + "_funcao_transformacao_eq.png", transformPlot);

    cout << "Processamento concluído para: " << inputPath << endl;

    if (computeLocalEq) {
        string eqDir = outputDir + "/EqualizacaoLocal";
        fs::create_directories(eqDir);
        Mat localEqImg = localHistogramEqualization(img);
        imwrite(eqDir + "/" + baseName + "_local_equalization.png", localEqImg);

        cout << "Equalização local concluída para: " << inputPath << endl;
    }
}

int main() {
    cout << "Iniciando o programa..." << endl;

    vector<string> inputImages = {"input/input1.tif", "input/input2.tif", "input/input3.tif"};

    string localEqImage = "input/input4.tif";
    string outputDir = "output";

    fs::create_directories(outputDir);

    for (const string &imgPath : inputImages) {
        processImage(imgPath, outputDir, false);
    }
    processImage(localEqImage, outputDir, true);

    cout << "Processo concluído para todas as imagens." << endl;
    return 0;
}
