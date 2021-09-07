//
//  BSDFgen.cpp
//  2Dsolve
//
//  Created by Mandy Xia.
//  Copyright © 2019 MandyXia. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include "MoM_ob.hpp"

#include <string.h>
#include <iomanip>

#include <cstdio>
#include <ctime>
#include <cstdlib>

#include "omp.h"
#include <Eigen/Dense>
#include <sys/time.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>

//Profiling
//#include <gperftools/profiler.h>

double hankelre[TABLE_SIZE];
double hankelim[TABLE_SIZE];
double dhankelre[TABLE_SIZE];
double dhankelim[TABLE_SIZE];

// save scattering disttribution, pdf, cdf as 32bit floats
void postprocess(bool lossless, Eigen::VectorXf& sigma, double energy, int phionum, int vectindex, float unit, Eigen::VectorXf& vect, Eigen::VectorXf& pdf, Eigen::VectorXf& cdf){
  double expected;
  if (lossless)
    expected = 1.f;
  else
    expected = (float) (energy + 1);

  if (expected<=0){
    vect.segment(vectindex, phionum).setZero();
    pdf.segment(vectindex, phionum).array() += (float) (1.0 / (2 * M_PI));
  }else{
    sigma *= expected / (sigma.sum() * unit);
    vect.segment(vectindex, phionum) = sigma;
    float normalization = sigma.sum() * unit;

    //std::cout << "Sigma total: " <<  sigma.sum() << std::endl;
    //std::cout << "Unit: " <<  unit << std::endl;
    //std::cout << "Normalization: " <<  normalization << std::endl;

    if (std::abs(normalization)<1e-6)
      pdf.segment(vectindex, phionum).array() = (float) (1.0 / (2 * M_PI));
    else
      pdf.segment(vectindex, phionum) = sigma / normalization;

  }
  for (int index = 0; index<phionum-1; ++index)
    cdf(vectindex+index) = pdf.segment(vectindex, index+1).sum() * unit;
  cdf(vectindex+phionum-1) = 1;
}

//New read 2D geometry function
std::vector<Eigen::Vector3d> readgeometry(std::string file){
  std::string line;
  std::ifstream myfile (file);

  std::vector<Eigen::Vector3d> nodes;

  if (myfile.is_open()){

    while ( getline (myfile,line) ){

      double x, y;
      std::stringstream stream_file(line);

      stream_file >> x;
      stream_file >> y;

      nodes.push_back(Eigen::Vector3d{x, y, 0});
    }
    
    myfile.close();
  }
  else{
    std::cout << "Unable to open file";
  }

  return nodes;
}

void readiorimag(std::vector<double> &kval, int num, std::string filename){
  std::cout<<"iorfile "<<filename<<std::endl;
  std::ifstream myfile(filename, std::ios::in|std::ios::binary);
  myfile.read((char*)&kval[0], num * sizeof(double));
}

int main(int argc, const char * argv[]) {
  std::string example = argv[1];
  std::string output = example+"/";
  std::cout<<"output directory "<<output<<std::endl;

  
  double radius, etare, etaim;
  int numel, phiinum, phionum, thetanum, lambdanum;
  std::vector<double> nval, kval;
  std::string filename, filename2, filename3;
  std::vector<Eigen::Vector3d> nodes;
  bool lossless = true;
  
  std::cout<<"------Simulating arbitrary cross-section------"<<std::endl;

  //Read geometry
  std::string cross_file = argv[2]; // filename of object's coordinates 
  std::cout<<"cross coords file: " << cross_file << std::endl;
  nodes = readgeometry(cross_file); // create nodes using the input file

  radius = -1; // setting the inital to be a negative number
  for (int i = 0; i < nodes.size(); ++i){
    double curradius = nodes[i].norm();
    if (radius < curradius)
      radius = curradius;
  }
  std::cout<<"radius "<<radius<<std::endl;
  std::cout<<"number of elements "<<nodes.size()<<std::endl;

  std::stringstream phionumstr(argv[3]); // number of outgoing phi directions
  phionumstr >> phionum;
  phiinum = phionum;

  std::stringstream thetanumstr(argv[4]); // number of theta directions
  thetanumstr >> thetanum;
  std::cout<<"phiinum "<<phiinum<<" phionum "<<phionum<<" thetanum "<<thetanum<<std::endl;

  std::stringstream lambdanumstr(argv[5]); // number of wavelength
  lambdanumstr >> lambdanum;
  std::cout<<"lambdanum "<<lambdanum<<std::endl;

  std::stringstream etarestr(argv[6]); // index of refraction of the fiber (real part)
  etarestr >> etare;

  //Wavelength dependent imaginary part of the IOR
  /*std::string iorfile = argv[8];
  kval.resize(lambdanum);
  readiorimag(kval, lambdanum, iorfile);*/

  /*for (auto im_ior: kval) {
    std::cout << im_ior << std::endl;
  }

  lossless = false;*/

  //Constant IOR
  std::stringstream etaimstr(argv[7]); // index of refraction of the fiber (imaginary part)
  etaimstr >> etaim;
  
  std::cout<<"etare "<<etare<<" etaim "<<etaim<<std::endl;

  if (etaim!=0)
    lossless = false;

  // wavelength parameter
  int lambdastart = 400;
  int lambdaend = 700;

  // read in hankel tables
  std::string hankeldir = "../hankeldouble/";
  filename = hankeldir + "hankelre.binary";
  std::ifstream myfile(filename, std::ios::in|std::ios::binary);
  myfile.read((char*)&hankelre[0], TABLE_SIZE * sizeof(double));

  filename = hankeldir + "hankelim.binary";
  std::ifstream myfile2(filename, std::ios::in|std::ios::binary);
  myfile2.read((char*)&hankelim[0], TABLE_SIZE * sizeof(double));

  filename = hankeldir + "dhankelre.binary";
  std::ifstream myfile3(filename, std::ios::in|std::ios::binary);
  myfile3.read((char*)&dhankelre[0], TABLE_SIZE * sizeof(double));

  filename = hankeldir + "dhankelim.binary";
  std::ifstream myfile4(filename, std::ios::in|std::ios::binary);
  myfile4.read((char*)&dhankelim[0], TABLE_SIZE * sizeof(double));

  int quadrature = 2;
  double mur = 1.0;
  double Dis = 1000*radius;
  char mode = 'M';
  double freq;
  std::complex<double> eta, epsr;

  struct timeval start, end;
  gettimeofday(&start, NULL);

  int nb_samples = thetanum * phiinum * phionum;
  std::cout<<"nb_samples "<<nb_samples<<std::endl; 
  int nb_pdf = thetanum * phiinum;
  float unit = (float) 2*M_PI / phionum;

  Eigen::VectorXf vect(nb_samples);
  Eigen::VectorXf pdf(nb_samples);
  Eigen::VectorXf cdf(nb_samples);
  Eigen::VectorXf cstot(nb_pdf * lambdanum);

  Eigen::VectorXf sigma;
  double energy;
  float cs;

  //ProfilerStart("simulation_profile.log");

  char dir_array[output.length()];
  strcpy(dir_array, output.c_str());
  mkdir(dir_array, 0777);

  // index of refraction calculation (global)  
  eta = etare - etaim * cunit;
  epsr = eta * eta;

  // wavelenght sampling: dense, random, sparse, etc
  int lambda_idx = 0;

  // dense sampling

  /*std::vector<double> lambdas = std::vector<double>(lambdanum);
  
  for (int i = 0; i < lambdanum; ++i){
     //double lambda = (lambdastart + (double)(lambdaend - lambdastart)/(double)lambdanum * i) * 1e-9;
     double lambda = (lambdastart + (double)(lambdaend - lambdastart)/(double)lambdanum * i);
     lambdas[i] = lambda;
  }*/

  // sparse sampling --> user defined
  /*lambdanum = 2;
  std::vector<double> lambdas { 400, 700 };*/

  // random sampling
  srand (time(NULL));//initialize random seed using the current pc time
  std::vector<double> lambdas = std::vector<double>(lambdanum);

  for (int i = 0; i < lambdanum; ++i){
     
     double lambda  = rand() % (lambdaend - lambdastart) + lambdastart;
     //lambda *= 1e-9; //Nano scale

     lambdas[i] = lambda;
  }


  for (double lambda : lambdas) {

    std::cout<<"lambda index: "<< lambda_idx << ", lambda: "<< lambda << " nm" << std::endl;

    lambda *= 1e-9; //Nano scale

    // Wavelength dependent imaginary part of the IOR
    //eta = etare - kval[i] * cunit;
    //epsr = eta * eta;

    freq = 299792458.0/double(lambda);
    MoM_ob m1;
    double theta;
    Eigen::PartialPivLU<Eigen::MatrixXcd> luvar;
    
    #pragma omp parallel for private (m1, theta, luvar)
    for (int j = 0 ; j < thetanum; ++j){
      theta = M_PI / 2 - M_PI / 2 * (double) j / (double) thetanum;

      m1 = MoM_ob(nodes, quadrature, 0, mode, freq, mur, epsr, theta);

      m1.assembly();
      luvar.compute(m1.Z);
      for (int k = 0; k < phiinum; ++k){
        
        double phi_i = (double) k / (double) phiinum * M_PI * 2;

        Eigen::VectorXf sigma1(phionum), sigma2(phionum);
        double energy1, energy2;

        m1.updatewave(phi_i, 'M', freq, theta);
        m1.incident();
        m1.sol = luvar.solve(m1.vvec);
        m1.BSDF(phionum, Dis, sigma1);
        
        if (lossless)
          energy1 = 0;
        else
          energy1= m1.energy();

        m1.updatewave(phi_i, 'E', freq, theta);
        m1.incident();
        m1.sol = luvar.solve(m1.vvec);
        m1.BSDF(phionum, Dis, sigma2);
        
        if (lossless)
          energy2 = 0;
        else
          energy2 = m1.energy();
        
        //std::cout<<"lossless: "<<lossless<<std::endl;

        sigma = (sigma1 + sigma2) * 0.5f;
        //sigma = sigma2;


        cs = (float) sigma.sum() / phionum * 2.f * M_PI;
        
        energy = (energy1 + energy2) / 2;
        //energy = energy2;

        cstot(lambda_idx*thetanum*phiinum + j*phiinum + k) = (float) (cs - energy);

        // process dist, cs, energy info for each wavelength and save to disk
        int vectindex = j*phiinum*phionum+k*phionum;
        postprocess(lossless, sigma, energy, phionum, vectindex, unit, vect, pdf, cdf);

      }
    }

    // write scattering distribution, pdf, cdf to disk
    filename = output + "TEM_"+std::to_string(lambda_idx)+".binary";
    std::ofstream out(filename, std::ios::out|std::ios::binary);
    out.write((char *) &vect(0), sizeof(float)*nb_samples);

    filename2 = output + "TEM_"+std::to_string(lambda_idx)+"_pdf.binary";
    std::ofstream out2(filename2, std::ios::out|std::ios::binary);
    out2.write((char *) &pdf(0), sizeof(float)*nb_samples);

    filename3 = output + "TEM_"+std::to_string(lambda_idx)+"_cdf.binary";
    std::ofstream out3(filename3, std::ios::out|std::ios::binary);
    out3.write((char *) &cdf(0), sizeof(float)*nb_samples);

     // update indices */
     lambda_idx++;
  }

  //ProfilerStop();
  
  // compute cross-section ratio and write to disc
  float bound = 6;
  float maxcs = std::min(bound, cstot.maxCoeff());
  std::cout<<"maxcs "<<maxcs<<std::endl;
  Eigen::VectorXf ratio = cstot / maxcs;
  // clamp ratio
  for (int i =0; i < nb_pdf * lambdanum; ++i){
    if (ratio(i) > 1)
      ratio(i) = 1;
    if(ratio(i) < 0){
      ratio(i) = 0;
      std::cout<<"ratio(i)<0, i "<<i<<" ratio(i) "<<ratio(i)<<std::endl;
    }
  }

  // output ratio
  filename = output + "ratio.binary";
  std::ofstream out(filename, std::ios::out|std::ios::binary|std::ios_base::app);
  out.write((char *) &ratio(0), sizeof(float)*nb_pdf*lambdanum);

  gettimeofday(&end, NULL);

  double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                  end.tv_usec - start.tv_usec) / 1.e6;
  std::cout<<"time used: "<<delta<<std::endl;

  return 0;
}
