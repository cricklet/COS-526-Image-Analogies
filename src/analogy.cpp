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

struct Features {
  ANNpointArray vectors; // double **
  int num;
  int dim;
  int offset;
  int width;
  int height;
};

Location GetFeatureLocation(int index, Features &features)
{
  Location l;
  l.i = (int) (index / features.height) + features.offset;
  l.j = index % features.height + features.offset;
  return l;
}

ANNpoint GetFeatureFromLocation(int aI, int aJ, Features &features)
{
  aI -= features.offset;
  aJ -= features.offset;
  return features.vectors[aI * features.height + aJ];
}

ANNpoint GetFeatureFromLocation(Location l, Features &features)
{
  return GetFeatureFromLocation(l.i, l.j, features);
}

double GetFeatureDist(ANNpoint v1, ANNpoint v2, int dim)
{
  double dist = 0;
  for (int i = 0; i < dim; i ++) {
    dist += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }
  return dist;
}

bool OutOfBounds(int i, int j, const R2Image *image)
{
  return i < 0 || j < 0 || i >= image->Width() || j >= image->Height();
}

bool OutOfBounds(int i, int j, Features &features, bool check_offset = true)
{
  int off = 0;
  if (check_offset) {
    off = features.offset;
  }
  return i < off || j < off
      || i >= features.width - off
      || j >= features.height - off;
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

static void
GetFeature(ANNpoint bFeature,
           int bI, int bJ, const R2Image *B, const R2Image *Bp,
           int region, bool partial)
{
  int dimA  = region * region;
  int dimAp = dimA;
  if (partial) {
    dimAp = floor(dimAp * 0.5);
  }
  GetConcattedVector(bI, bJ, B, Bp, region, bFeature, dimA, dimAp);
}

static Features
GetFeatures(const R2Image *A, const R2Image *Ap,
            int region, bool partial, int offset)
{
  int dimA  = region * region;
  int dimAp = dimA;
  if (partial) {
    dimAp = floor(dimAp * 0.5);
  }

  int dim = dimA + dimAp;

  int AWidth = A->Width() - 2 * offset;
  int AHeight = A->Height() - 2 * offset;

  int num = AWidth * AHeight;

  ANNpointArray vectors = annAllocPts(num, dim);

  for (int aI = 0; aI < AWidth; aI++) {
    for (int aJ = 0; aJ < AHeight; aJ++) {
      int index  = (aI * AHeight + aJ);

      GetConcattedVector(aI + offset, aJ + offset, A, Ap, region,
                         vectors[index], dimA, dimAp);
    }
  }

  Features features;
  features.dim = dim;
  features.offset = offset;
  features.width = AWidth;
  features.height = AHeight;
  features.num = num;
  features.vectors = vectors;

  return features;
}

////////////////////////////////////////////////////////////////////////////////
// Brute force methods

static int
FindBestMatchBrute(Features features, ANNpoint query) {
  double minDist = DBL_MAX;
  double minIndex;

  for (int i = 0; i < features.num; i ++) {
    double dist = GetFeatureDist(features.vectors[i], query, features.dim);
    if (dist < minDist) {
      minDist = dist;
      minIndex = i;
    }
  }
  return minIndex;
}

static Location
FindBestMatchBrute(int bI, int bJ, const R2Image *A, const R2Image *Ap, const R2Image *B, R2Image *Bp) {
  int region = 5;

  Features features = GetFeatures(A, Ap, region, true, 2);

  ANNpoint bFeature = annAllocPt(features.dim);
  GetFeature(bFeature, bI, bJ, B, Bp, region, true);

  int index = FindBestMatchBrute(features, bFeature);
  Location loc = GetFeatureLocation(index, features);

  annDeallocPt(bFeature);
  delete features.vectors;
  return loc;
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
// ANN methods

static int
ANNMatch(ANNpoint queryPoint, ANNkd_tree *kdTree)
{
  int numNN = 1;
  ANNidx nnIdx;
  ANNdist nnDist;
  int errorBound = 0;

  kdTree->annkSearch(queryPoint, numNN, &nnIdx, &nnDist, errorBound);

  return nnIdx;
}

static Location
CoherenceMatch(int bI, int bJ,
               ANNpoint bFeature,
               Features &features,
               Location **sources)
{
  double minDist = DBL_MAX;
  Location minLoc;

  for (int dJ = -1; dJ <= 0; dJ ++) {
    for (int dI = -1; dI <= 1; dI ++) {
      // Skip parts of Bp that haven't been rendered yet
      if (dJ > 0) break;
      if (dJ == 0 && dI >= 0) break;

      if (OutOfBounds(bI + dI, bJ + dJ, features, false)) {
        continue;
      }

      Location a = sources[bI + dI][bJ + dJ];
      int aI = a.i - dI;
      int aJ = a.j - dJ;

      if (OutOfBounds(aI, aJ, features)) {
        continue;
      }

      ANNpoint aFeature = GetFeatureFromLocation(aI, aJ, features);
      double dist = GetFeatureDist(aFeature, bFeature, features.dim);
      if (dist < minDist) {
        minDist = dist;
        minLoc.i = aI;
        minLoc.j = aJ;
      }
    }
  }

  return minLoc;
}

static void
RunAnalogyANN(const R2Image *A, const R2Image *Ap,
              const R2Image *B, R2Image *Bp,
              double coherence, Location **sources)
{
  int region = 5;
  int Aoffset = 2;
  Features features = GetFeatures(A, Ap, region, true, Aoffset);
  ANNkd_tree *kdTree = new ANNkd_tree(features.vectors, features.num, features.dim);
  ANNpoint bFeature = annAllocPt(features.dim);

  printf("Generating Bp\n");
  for (int bI = 0; bI < B->Width(); bI++) {
    printf("%d out of %d\n", bI, B->Width());
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      // Record the source of the pixel
      GetFeature(bFeature, bI, bJ, B, Bp, region, true);
      int annIndex = ANNMatch(bFeature, kdTree);

      Location annLoc = GetFeatureLocation(annIndex, features);
      Location cohLoc = CoherenceMatch(bI, bJ, bFeature, features, sources);

      ANNpoint annFeature = GetFeatureFromLocation(annLoc, features);
      ANNpoint cohFeature = GetFeatureFromLocation(cohLoc, features);

      double annDist = GetFeatureDist(annFeature, bFeature, features.dim);
      double cohDist = GetFeatureDist(cohFeature, bFeature, features.dim);

      int aI, aJ;
      if (cohDist <= annDist * (1 + pow(2, 0) * coherence)) {
        aI = cohLoc.i;
        aJ = cohLoc.j;
        sources[bI][bJ] = cohLoc;
      } else {
        aI = annLoc.i;
        aJ = annLoc.j;
        sources[bI][bJ] = annLoc;
      }

      R2Pixel p;
      R2Pixel pixelAp = Ap->Pixel(aI, aJ);
      R2Pixel pixelB = B->Pixel(bI, bJ);
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);
    }
  }

  annDeallocPt(bFeature);
  delete features.vectors;
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
  RunAnalogyANN(newA, newAp, B, Bp, 0.2, sources);

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
