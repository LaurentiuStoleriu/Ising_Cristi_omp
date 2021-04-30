#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <random>
#include <iostream>
#include <memory>

#include <omp.h>

using std::cout; using std::endl; using std::unique_ptr;

#define N1 1000
#define N1d2 (N1/2)
#define N2 1000

#define VOLUME N1*N2
#define VOLUMEd2 N1d2*N2

/* 2d Ising model using Metropolis update algorithm */

int main(int argc, char **argv)
{
	int n, i, j, iter;
	//int xup[VOLUME], yup[VOLUME], xdn[VOLUME], ydn[VOLUME];
	unique_ptr<int[]>xup(new int[VOLUME]);
	unique_ptr<int[]>yup(new int[VOLUME]);
	unique_ptr<int[]>xdn(new int[VOLUME]);
	unique_ptr<int[]>ydn(new int[VOLUME]);

	float beta, esum, mag;
	//float s[VOLUME];
	unique_ptr<float[]>s(new float[VOLUME]);

	double esumt, magt;

	/* Parameters. Beta is the inverse of the temperature */
	beta = 0.5;
	iter = 500;

	/* Set random seed  */
	std::mt19937 generator(123);
	std::uniform_real_distribution<double> uniform_dstrb(0.0, 1.0);

    //clock_t tic = clock();
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);

	/* Initialize each point randomly */
	for (i = 0; i < VOLUME; i++)
	{
		if (uniform_dstrb(generator) < 0.5)
			s[i] = 1.0;
		else
			s[i] = -1.0;
	}

	/* Create and index of neighbours
	   The sites are partitioned to even an odd,
	   with even sites first in the array */
	for (j = 0; j < N2; j++)
		for (i = 0; i < N1; i++)
		{
			int i2 = i / 2;
			int is = i2 + j * N1d2;
			if ((i + j) % 2 == 0)
			{
				xup[is] = ((i + 1) / 2) % N1d2 + j * N1d2 + VOLUMEd2;
				yup[is] = i2 + ((j + 1) % N2) * N1d2 + VOLUMEd2;
				xdn[is] = ((i - 1 + N1) / 2) % N1d2 + j * N1d2 + VOLUMEd2;
				ydn[is] = i2 + ((j - 1 + N2) % N2) * N1d2 + VOLUMEd2;
			}
			else
			{
				xup[is + VOLUMEd2] = ((i + 1) / 2) % N1d2 + j * N1d2;
				yup[is + VOLUMEd2] = i2 + ((j + 1) % N2) * N1d2;
				xdn[is + VOLUMEd2] = ((i - 1 + N1) / 2) % N1d2 + j * N1d2;
				ydn[is + VOLUMEd2] = i2 + ((j - 1 + N2) % N2) * N1d2;
			}
		}

	/* Initialize the measurements */
	esumt = 0.0;
	magt = 0.0;

	/* Run a number of iterations */
	for (n = 0; n < iter; n++)
	{
		esum = 0.0;
		mag = 0.0;

		/* Loop over the lattice and try to flip each atom */
#pragma omp parallel for reduction( +: esum, mag )
		for (i = 0; i < VOLUME; i++)
		{
			float new_energy, energy_now, deltae;
			float stmp;
			float neighbours = s[xup[i]] + s[yup[i]] + s[xdn[i]] + s[ydn[i]];
			stmp = -s[i];

			/* Find the energy before and after the flip */
			energy_now = -s[i] * neighbours;
			new_energy = -stmp * neighbours;
			deltae = new_energy - energy_now;

			/* Accept or reject the change */
			if (exp(-beta * deltae) > uniform_dstrb(generator)) {
				s[i] = stmp;
				energy_now = new_energy;
			}

			/* Measure magnetisation and energy */
			mag = mag + s[i];
			esum = esum + energy_now;
		}

		/* Calculate measurements and add to run averages  */
		esum = esum / (VOLUME);
		mag = mag / (VOLUME);
		esumt = esumt + esum;
		magt = magt + fabs(mag);

		cout << "average energy = " << esum << ", average magnetization = " << mag << endl;
	}

	esumt = esumt / iter;
	magt = magt / iter;
	cout << "Over the whole simulation:" << endl;
	cout << "average energy = " << esumt << ", average magnetization = " << magt << endl;

    gettimeofday(&toc, NULL);
    double delta_t = ((toc.tv_sec  - tic.tv_sec) * 1000000u + toc.tv_usec - tic.tv_usec) / 1.0e6;
    cout << "ran for " << delta_t << " seconds" << endl;

	return 0;
}