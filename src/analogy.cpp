// COS 526, Spring 2010, Assignment 1



// Include files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <climits>
#include <ANN/ANN.h>
#include "R2Pixel.h"
#include "R2Image.h"

// Program arguments

static char *A_filename = NULL;
static char *Ap_filename = NULL;
static char *B_filename = NULL;
static char *Bp_filename = NULL;
static R2Pixel mask_color = R2blue_pixel;
static int neighborhood_size = 5;

////////////////////////////////////////////////////////////////////////////////

struct Location {
  int i; int j;
};

bool OutOfBounds(int i, int j, const R2Image *image)
{
  return i < 0 || j < 0 || i >= image->Width() || j >= image->Height();
}

double GetLuminosity(int i, int j, const R2Image *image)
{
  return image->Pixel(i, j).Y();
}

double Gauss(int dI, int dJ, int regionSize)
{
  double distSq = dI * dI + dJ * dJ;
  double sigmaSq = regionSize * 0.4;
  double gauss = exp(- distSq / sigmaSq);
  return gauss;
  // return 1;
}

static double
CalculateMeanLuminance(const R2Image *A)
{
  double lumSum = 0;
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      lumSum += GetLuminosity(aI, aJ, A);
    }
  }

  return lumSum / (A->Width() * A->Height());
}

static double
CalculateStdDev(const R2Image *A, double mean)
{
  double variance = 0;
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      double diff = GetLuminosity(aI, aJ, A) - mean;
      variance += diff * diff;
    }
  }

  return sqrt(variance / (A->Width() * A->Height()));
}

static void
RemapLuminance(R2Image *A, double meanA, double stdDevA, double meanB, double stdDevB)
{
  printf("%f, %f, %f, %f\n", meanA, stdDevA, meanB, stdDevB);
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      R2Pixel o = A->Pixel(aI, aJ);
      double newY = (stdDevB / stdDevA) * (o.Y() - meanA) + meanB;

      R2Pixel p;
      p.SetYIQ(newY, o.I(), o.Q());
      A->SetPixel(aI, aJ, p);
    }
  }
}

static void
RemapLuminance(R2Image *A, R2Image *Ap, const R2Image *B)
{
  double meanA = CalculateMeanLuminance(A);
  double meanB = CalculateMeanLuminance(B);

  double stdDevA = CalculateStdDev(A, meanA);
  double stdDevB = CalculateStdDev(B, meanB);

  RemapLuminance(A,  meanA, stdDevA, meanB, stdDevB);
  RemapLuminance(Ap, meanA, stdDevA, meanB, stdDevB);
}

////////////////////////////////////////////////////////////////////////////////
// ANN methods

static void
GetVector(int aI, int aJ, const R2Image *A, int regionSize,
          double *vector, int dimension)
{
  for (int regionI = 0; regionI < regionSize; regionI ++) {
    for (int regionJ = 0; regionJ < regionSize; regionJ ++) {
      int regionIndex = regionI * regionSize + regionJ;
      int dI = regionI - regionSize / 2;
      int dJ = regionJ - regionSize / 2;

      // Skip if this is out of bounds
      if (OutOfBounds(aI + dI, aJ + dJ, A)) {
        continue;
      }

      // Skip if the relevant pixel doesn't exist yet.
      if (regionIndex >= dimension) {
        continue;
      }

      vector[regionIndex] = GetLuminosity(aI + dI, aJ + dJ, A) * Gauss(dI, dJ, regionSize);
    }
  }
}

static void
RunAnalogyANN(const R2Image *A, const R2Image *Ap,
              const R2Image *B, R2Image *Bp)
{
  int regionSize = 5;
  int regionDim = regionSize * regionSize;
  int imagePixels = A->Width() * A->Height();

  int regionDimPartial = ceil(regionDim * 0.5);
  int searchDimensions = regionDim + regionDimPartial;

  printf("Caching all vectors from A, Ap\n");

  // Cache all the A and Ap feature vectors
  ANNpointArray vectorsA;
  vectorsA = annAllocPts(imagePixels, searchDimensions);
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      int index  = (aI * A->Height() + aJ);

      GetVector(aI, aJ, A,  regionSize, &vectorsA[index][0], regionDim);
      GetVector(aI, aJ, Ap, regionSize, &vectorsA[index][regionDim], regionDimPartial);
    }
  }

  printf("Generating kd tree\n");

  ANNkd_tree *kdTree = new ANNkd_tree(vectorsA, imagePixels, searchDimensions);

  int numNN = 1; // nearest neighbors
  ANNidxArray nnIdx = new ANNidx[numNN];
  ANNdistArray nnDists = new ANNdist[numNN];
  double errorBound = 0;

  ANNpoint queryPoint = annAllocPt(searchDimensions);

  printf("Generating Bp\n");
  for (int bI = 0; bI < B->Width(); bI++) {
    printf("%d out of %d\n", bI, B->Width());
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      GetVector(bI, bJ, B,  regionSize, &queryPoint[0], regionDim);
      GetVector(bI, bJ, Bp, regionSize, &queryPoint[regionDim], regionDimPartial);

      // for (int i = 0; i < searchDimensions; i ++) {
      //   printf("%f ", queryPoint[i]);
      // }
      // printf("\n");

      kdTree->annkSearch(queryPoint, numNN, nnIdx, nnDists, errorBound);

      int aI = (int) (nnIdx[0] / A->Height());
      int aJ = nnIdx[0] % A->Height();

      // printf("%d, %d\n", aI, aJ);

      R2Pixel pixelAp = Ap->Pixel(aI, aJ);
      R2Pixel pixelB = B->Pixel(bI, bJ);

      R2Pixel p;
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);
    }
  }

  delete kdTree;
  annClose();
}

////////////////////////////////////////////////////////////////////////////////
// Brute force methods

double CalculateDistanceBrute(int aI, int aJ, int bI, int bJ,
  const R2Image *A, const R2Image *Ap, const R2Image *B, R2Image *Bp,
  int regionSize)
{
  double distance = 0;

  for (int regionI = 0; regionI < regionSize; regionI ++) {
    for (int regionJ = 0; regionJ < regionSize; regionJ ++) {
      int dI = regionI - regionSize / 2;
      int dJ = regionJ - regionSize / 2;

      // Skip if this is out of bounds
      if (OutOfBounds(aI + dI, aJ + dJ, A) ||
          OutOfBounds(bI + dI, bJ + dJ, B)) {
         continue;
      }

      // Get all relevant values
      double valA  = GetLuminosity(aI + dI, aJ + dJ, A);
      double valAp = GetLuminosity(aI + dI, aJ + dJ, Ap);
      double valB  = GetLuminosity(aI + dI, aJ + dJ, B);
      double valBp = GetLuminosity(aI + dI, aJ + dJ, Bp);

      // Take a squared difference of them
      distance += (valB - valA) * (valB - valA);

      // Skip Bp if the relevant pixel doesn't exist yet.
      if (dJ > 0) continue;
      if (dJ == 0 && dI >= 0) continue;

      distance += (valBp - valAp) * (valBp - valAp);
    }
  }

  return distance;
}

Location FindBestMatchBrute(int bI, int bJ, const R2Image *A, const R2Image *Ap, const R2Image *B, R2Image *Bp) {
  int region = 5;
  Location p;

  double minDist = INT_MAX;
  int minAI = -1;
  int minAJ = -1;

  // Loop through A, storing the best match aI, aJ for region bI, bJ
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      double dist = CalculateDistanceBrute(aI, aJ, bI, bJ, A, Ap, B, Bp, region);
      if (dist < minDist) {
        minAI = aI;
        minAJ = aJ;
      }
    }
  }

  p.i = minAI;
  p.j = minAJ;

  return p;
}

static void
RunAnalogyBrute(const R2Image *A, const R2Image *Ap,
                const R2Image *B, R2Image *Bp)
{
  for (int bI = 0; bI < B->Width(); bI++) {
    printf("%d out of %d\n", bI, B->Width());
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      Location a = FindBestMatchBrute(bI, bJ, A, Ap, B, Bp);
      int aI = a.i;
      int aJ = a.j;

      R2Pixel pixelB = B->Pixel(bI, bJ);
      R2Pixel pixelAp = Ap->Pixel(aI, aJ);

      R2Pixel p;
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper code

static R2Image *
CreateAnalogyImage(const R2Image *A, const R2Image *Ap, const R2Image *B)
{
  // Allocate Bp image
  R2Image *Bp = new R2Image(B->Width(), B->Height());
  if (!Bp) {
    fprintf(stderr, "Unable to allocate output_image\n");
    exit(-1);
  }

  R2Image *newA  = new R2Image(*A);
  R2Image *newAp = new R2Image(*Ap);
  RemapLuminance(newA, newAp, B);

  // RunAnalogyBrute(newA, newAp, B, Bp);
  RunAnalogyANN(newA, newAp, B, Bp);

  // Return Bp image
  return Bp;
}



static R2Image *
ReadImage(const char *filename)
{
  // Allocate input_image
  R2Image *image = new R2Image();
  if (!image) {
    fprintf(stderr, "Unable to allocate image\n");
    return NULL;
  }

  // Read input image
  if (!image->Read(filename)) {
    fprintf(stderr, "Unable to read image from %s\n", filename);
    return NULL;
  }

  // Return image
  return image;
}

static int
ParseArgs(int argc, char **argv)
{
  // Parse program arguments
  argv++, argc--;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-neighborhood_size")) { argv++; argc--; neighborhood_size = atoi(*argv); }
      else if (!strcmp(*argv, "-mask_color")) { mask_color.Reset(atof(argv[1]),atof(argv[2]),atof(argv[3]),1); argv+=3; argc-=3; }
      else { fprintf(stderr, "Invalid option: %s\n", *argv); return 0; }
    }
    else {
      if (!A_filename) A_filename = *argv;
      else if (!Ap_filename) Ap_filename = *argv;
      else if (!B_filename) B_filename = *argv;
      else if (!Bp_filename) Bp_filename = *argv;
      else { fprintf(stderr, "Invalid option: %s\n", *argv); return 0; }
    }
    argv++, argc--;
  }

  // Check program arguments
  if (!A_filename || !Ap_filename || !B_filename || !Bp_filename) {
    fprintf(stderr, "Usage: analogy A A\' B B\'[-mask_color r g b] [-neighborhood_size npixels]\n");
    return 0;
  }

  // Return success
  return 1;
}



int
main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read input images
  R2Image *A = ReadImage(A_filename);
  if (!A) exit(-1);
  R2Image *Ap = ReadImage(Ap_filename);
  if (!Ap) exit(-1);
  R2Image *B = ReadImage(B_filename);
  if (!B) exit(-1);

  // Create output image
  R2Image *Bp = CreateAnalogyImage(A, Ap, B);
  if (!Bp) exit(-1);

  // Write output image
  if (!Bp->Write(Bp_filename)) exit(-1);

  // Return success
  return 0;
}
