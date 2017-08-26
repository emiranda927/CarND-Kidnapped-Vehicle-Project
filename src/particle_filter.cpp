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
	num_particles = 150;
	//generate seed for random number generator
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine gen(seed);

	//create normal distributions for sampling init values
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0; i<num_particles; ++i){
		double  sample_x, sample_y, sample_theta;
		Particle p; //create particle instance

		//sample init values from normal distribution
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		//assign values to particle and push onto tracked list
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1./num_particles;
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
	default_random_engine gen(seed);

	for (int i=0; i<num_particles; ++i){
		double x_p = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta+(yaw_rate*delta_t))-sin(particles[i].theta));
		double y_p = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+(yaw_rate*delta_t)));
		double theta_p = particles[i].theta + (yaw_rate*delta_t);

		normal_distribution<double> dist_x_p(x_p, std_pos[0]);
		normal_distribution<double> dist_y_p(y_p, std_pos[1]);
		normal_distribution<double> dist_theta_p(theta_p, std_pos[2]);

		particles[i].x = dist_x_p(gen);
		particles[i].y = dist_y_p(gen);
		particles[i].theta = dist_theta_p(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i<observations.size(); i++){

		double min_distance = 999.;
		int id = -1;
		LandmarkObs obs = observations[i];

		for (int j=0; j<predicted.size(); j++){

			LandmarkObs pred = predicted[i];
			double distance = dist(obs.x, obs.y, pred.x, pred.y);

			if (distance <= min_distance){
				min_distance = distance;
				id = pred.id;
			}
		}

		observations[i].id = id;
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

	/*for each map landmark
	* get id and coords for landmarks within sensor range
	* then add landmark data to prediction vector.
	* create a vector to store transformed observations from observations
	* then perform a data association of the predictions in-range and trans. obs.
	* calculate MV gaussian on associated predictions and transformed obs
	* update weight of particle
	*/

	for (int i=0; i<num_particles; i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> landmarks_in_range;
		vector<LandmarkObs> observations_m;
		double px_m, py_m;

		for(int j=0; j<map_landmarks.landmark_list.size(); j++){
			float x_m = map_landmarks.landmark_list[j].x_f;
			float y_m = map_landmarks.landmark_list[j].y_f;
			int id_m = map_landmarks.landmark_list[j].id_i;

			if(dist(p_x, p_y, x_m, y_m) <= sensor_range){
				landmarks_in_range.push_back(LandmarkObs{id_m, x_m, y_m});
			}
		}

		for(int k=0; k<observations.size(); k++){
			px_m = p_x + (cos(p_theta) * observations[k].x) - (sin(p_theta) * observations[k].y);
			py_m = p_y + (sin(p_theta) * observations[k].x) + (cos(p_theta) * observations[k].y);
			observations_m.push_back(LandmarkObs{observations[k].id, px_m, py_m});
		}

		dataAssociation(landmarks_in_range, observations_m);

		for (int l=0; l<observations_m.size(); l++){ //for each observation
		  int obs_id = observations_m[l].id; //grab id
		  double predict_x, predict_y;

		  for (int m=0; m<landmarks_in_range.size(); m++){
		    if (landmarks_in_range[m].id == obs_id){
		      predict_x = landmarks_in_range[m].x;
		      predict_y = landmarks_in_range[m].y;
		    }
		  }

			//calculate weights
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double c = 1./(2.*M_PI*std_x*std_y);
			double upper = (pow((predict_x - px_m),2)/(2.*std_x*std_x))+(pow((predict_y-py_m),2)/(2.*std_y*std_y));
			float p = c*exp(-upper);

			particles[i].weight *= p;
		}


	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine gen (seed);
  vector<Particle> new_particles;

	  // get all of the current weights
	  vector<double> weights;
	  for (int i = 0; i < num_particles; i++) {
	    weights.push_back(particles[i].weight);
	  }

	  // generate random starting index for resampling wheel
	  uniform_int_distribution<int> uniintdist(0, num_particles-1);
	  auto index = uniintdist(gen);

	  // get max weight
	  double max_weight = *max_element(weights.begin(), weights.end());

	  // uniform random distribution [0.0, max_weight)
	  uniform_real_distribution<double> unirealdist(0.0, max_weight);

	  double beta = 0.0;

	  // spin the resample wheel!
	  for (int i = 0; i < num_particles; i++) {
	    beta += unirealdist(gen) * 2.0;
	    while (beta > weights[index]) {
	      beta -= weights[index];
	      index = (index + 1) % num_particles;
	    }
	    new_particles.push_back(particles[index]);
	  }

	  particles = new_particles;

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
