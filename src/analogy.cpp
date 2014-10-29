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
static char *Bs_filename = NULL;
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

      // Skip if the relevant pixel doesn't exist yet.
      if (regionIndex >= dimension) {
        continue;
      }

      // Skip if this is out of bounds
      if (OutOfBounds(aI + dI, aJ + dJ, A)) {
        vector[regionIndex] = 0;
        continue;
      }

      vector[regionIndex] = GetLuminosity(aI + dI, aJ + dJ, A) * Gauss(dI, dJ, regionSize);
    }
  }
}

static void
GetConcattedVector(int aI, int aJ, const R2Image *A, const R2Image *Ap,
                   int region, double *vector, int dim1, int dim2)
{
  GetVector(aI, aJ, A,  region, &vector[0], dim1);
  GetVector(aI, aJ, Ap, region, &vector[dim1], dim2);
}

class ANNSearch {
public:
  ANNSearch(const R2Image *A, const R2Image *Ap);
  ~ANNSearch();
  Location Query(int bI, int bJ, const R2Image *B, R2Image *Bp);
  void GetVector(int bI, int bJ, const R2Image *Bp);

private:
  ANNkd_tree *kdTree;
  int numNN;
  ANNidx *nnIdx;
  ANNdist *nnDists;
  int errorBound;
  ANNpoint queryPoint;
  int regionSize;

  int regionDim;
  int regionDimPartial;
  int offset;
  int AWidth;
  int AHeight;

  // I'd much prefer to avoid using OOP here and generate anonymous functions
  // instead, but I don't know how to do that in C++...
};

ANNSearch::
~ANNSearch()
{
  annClose();
  delete this->kdTree;
  delete this->nnIdx;
  delete this->nnDists;
  annDeallocPt(this->queryPoint);
}

ANNSearch::
ANNSearch(const R2Image *A, const R2Image *Ap)
{
  // Search parameters
  this->regionSize = 5;
  this->regionDim = regionSize * regionSize;
  this->regionDimPartial = floor(regionDim * 0.5);
  this->offset = regionSize / 2;

  int searchDim = regionDim + regionDimPartial;

  printf("Generating kd tree of vectors from A & Ap\n");
  this->AWidth = A->Width() - 2 * offset;
  this->AHeight = A->Height() - 2 * offset;

  int numVectors = (A->Width() - 2 * offset) * (A->Height() - 2 * offset);

  ANNpointArray vectorsA = annAllocPts(numVectors, searchDim);
  for (int aI = 0; aI < AWidth; aI++) {
    for (int aJ = 0; aJ < AHeight; aJ++) {
      int index  = (aI * AHeight + aJ);

      GetConcattedVector(aI + offset, aJ + offset, A, Ap, regionSize,
          &vectorsA[index][0], regionDim, regionDimPartial);
    }
  }

  // Prep kd tree
  this->kdTree = new ANNkd_tree(vectorsA, numVectors, searchDim);

  // Prep nearest neighbor search
  this->numNN = 1;
  this->nnIdx = new ANNidx[numNN];
  this->nnDists = new ANNdist[numNN];
  this->errorBound = 0;
  this->queryPoint = annAllocPt(searchDim);

  annDeallocPts(vectorsA);
}

Location ANNSearch::
Query(int bI, int bJ, const R2Image *B, R2Image *Bp)
{
  GetConcattedVector(bI, bJ, B, Bp, regionSize,
      &queryPoint[0], regionDim, regionDimPartial);
  kdTree->annkSearch(queryPoint, numNN, nnIdx, nnDists, errorBound);

  Location l;
  l.i = (int) (nnIdx[0] / AHeight) + offset;
  l.j = nnIdx[0] % AHeight + offset;

  return l;
}

static void
RunAnalogyANN(const R2Image *A, const R2Image *Ap,
              const R2Image *B, R2Image *Bp,
              Location **sources)
{
  ANNSearch *searcher = new ANNSearch(A, Ap);

  printf("Generating Bp\n");
  for (int bI = 0; bI < B->Width(); bI++) {
    printf("%d out of %d\n", bI, B->Width());
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      // Record the source of the pixel
      Location l = searcher->Query(bI, bJ, B, Bp);

      int aI = l.i;
      int aJ = l.j;
      sources[bI][bJ] = l;

      R2Pixel p;
      R2Pixel pixelAp = Ap->Pixel(aI, aJ);
      R2Pixel pixelB = B->Pixel(bI, bJ);
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);
    }
  }

  delete searcher;
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
      double valWeight = Gauss(dI, dJ, regionSize);
      double valA  = GetLuminosity(aI + dI, aJ + dJ, A)  * valWeight;
      double valAp = GetLuminosity(aI + dI, aJ + dJ, Ap) * valWeight;
      double valB  = GetLuminosity(bI + dI, bJ + dJ, B)  * valWeight;
      double valBp = GetLuminosity(bI + dI, bJ + dJ, Bp) * valWeight;

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

  double minDist = DBL_MAX;
  int minAI = -1;
  int minAJ = -1;

  // Loop through A, storing the best match aI, aJ for region bI, bJ
  for (int aI = 3; aI < A->Width() - 3; aI++) {
    for (int aJ = 3; aJ < A->Height() - 3; aJ++) {
      double dist = CalculateDistanceBrute(aI, aJ, bI, bJ, A, Ap, B, Bp, region);
      if (dist < minDist) {
        minAI = aI;
        minAJ = aJ;
        minDist = dist;
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

  Location **sources = new Location *[Bp->Width()];
  for (int i = 0; i < Bp->Width(); i ++) {
    sources[i] = new Location[Bp->Height()];
  }

  // RunAnalogyBrute(newA, newAp, B, Bp);
  RunAnalogyANN(newA, newAp, B, Bp, sources);

  R2Image *sourcesImage = new R2Image(*Bp);
  for (int bI = 0; bI < B->Width(); bI++) {
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      Location a = sources[bI][bJ];
      double aI = a.i;
      double aJ = a.j;

      R2Pixel p;
      p.SetRed(aI / B->Width());
      p.SetGreen(aJ / B->Height());
      sourcesImage->SetPixel(bI, bJ, p);
    }
  }

  sourcesImage->Write(Bs_filename);

  delete sourcesImage;
  delete newA;
  delete newAp;
  for (int i = 0; i < Bp->Width(); i ++) delete sources[i];
  delete sources;

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

  Bs_filename = new char[strlen(Bp_filename) + 4];
  strcpy(Bs_filename, Bp_filename);
  strstr(Bs_filename, ".bmp")[0] = '\0';
  strcat(Bs_filename, ".src.bmp");
  printf("%s\n", Bs_filename);

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
