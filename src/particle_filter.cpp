/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <chrono>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	weights.resize(num_particles, 1.0);
	//generate seed for random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine gen (seed);

	//create normal distributions for sampling init values
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  //sample init values from normal distribution
  double sample_x = dist_x(gen);
  double sample_y = dist_y(gen);
  double sample_theta = dist_theta(gen);

	for (int i=0; i<num_particles; i++){
		Particle p; //create particle instance

    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1.;
    particles.push_back(p);
	}
	is_initialized = true; //set init flag to true
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine gen (seed);

	normal_distribution<double> dist_x_p(0, std_pos[0]);
	normal_distribution<double> dist_y_p(0, std_pos[1]);
	normal_distribution<double> dist_theta_p(0, std_pos[2]);

	for (int i=0; i<num_particles; i++){
	  if (fabs(yaw_rate) < 0.0001) {
	      yaw_rate = 0.0001;
	    }
    particles[i].x += (velocity / yaw_rate) * ( sin( particles[i].theta + yaw_rate*delta_t ) - sin(particles[i].theta) );
    particles[i].y += (velocity / yaw_rate) * ( cos( particles[i].theta ) - cos( particles[i].theta + yaw_rate*delta_t ) );
    particles[i].theta += yaw_rate * delta_t;

	  particles[i].x += dist_x_p(gen);
		particles[i].y += dist_y_p(gen);
		particles[i].theta += dist_theta_p(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i<observations.size(); i++){

		double min_distance = 10000.0;

		for (int j=0; j<predicted.size(); j++){

			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance <= min_distance){
				min_distance = distance;
				observations[i].id = j;
			}
		}
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double c = 1./(2.*M_PI*std_x*std_y);

	for (int i=0; i<particles.size(); i++){
	  double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		particles[i].weight = 1./num_particles;

		vector<LandmarkObs> landmarks_in_range;
		vector<LandmarkObs> observations_m;
		double px_m, py_m;

		//Taking all map landmarks and creating a vector with landmarks in range of particle
		for(int j=0; j<map_landmarks.landmark_list.size(); j++){
			float x_m = map_landmarks.landmark_list[j].x_f;
			float y_m = map_landmarks.landmark_list[j].y_f;
			int id_m = map_landmarks.landmark_list[j].id_i;

			if(dist(p_x, p_y, x_m, y_m) <= sensor_range){
				landmarks_in_range.push_back(LandmarkObs{id_m, x_m, y_m});
			}
		}

		//transform observations into map coordinate system
    for(int k=0; k<observations.size(); k++){
      px_m = p_x + (cos(p_theta) * observations[k].x) - (sin(p_theta) * observations[k].y);
      py_m = p_y + (sin(p_theta) * observations[k].x) + (cos(p_theta) * observations[k].y);
      observations_m.push_back(LandmarkObs{observations[k].id, px_m, py_m});
    }

    dataAssociation(landmarks_in_range, observations_m);
    int obs_id;
    double predict_x, predict_y, lm_x, lm_y;
    double p_weight = 1.;

    for (int l=0; l<observations_m.size(); l++){ //for each observation
      obs_id = observations_m[l].id; //grab id
      predict_x = observations_m[l].x;
      predict_y = observations_m[l].y;

      lm_x = landmarks_in_range[obs_id].x;
      lm_y = landmarks_in_range[obs_id].y;

      //calculate weights
      double upper = (pow((predict_x - lm_x),2)/(2.*std_x*std_x))+(pow((predict_y-lm_y),2)/(2.*std_y*std_y));
      double p = c*exp(-upper);
      p_weight *= p;
    }
    particles[i].weight = p_weight;
    weights[i] = p_weight;
	}
}

void ParticleFilter::resample() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine gen (seed);

  discrete_distribution<> dist(weights.begin(), weights.end());
  vector<Particle> resampled;

  for (size_t i = 0; i < particles.size(); ++i) {
    const Particle &p = particles[dist(gen)];
    resampled.push_back(p);
  }
  particles.clear();
  particles.insert(particles.end(), resampled.begin(), resampled.end());
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
