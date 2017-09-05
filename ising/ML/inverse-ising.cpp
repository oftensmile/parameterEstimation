//  Two-dimensional Potts Model
// iAskin , Juius; Teller , Edward(1943). "Statistics of Two-Dimensional Lattices With Four Componets"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
using namespace std;

inline double std_rand(){ return rand() / (RAND_MAX + 1.0); }

int L = 10;
int N = L*L;                          // number of spins in x and y
int n_q = 2;                          // number of state 
// S_i = 0, 1,.., n_q-1
int n_sample = 1000; // iteration number of the optimaisation.
int epoc_stop = 500; // iteration number of the optimaisation.
double lr = 0.2;	// learning rate of Gradient Decent
double regula = 1/n_sample;	// 
double T=1.0;                           // temperature in units of k_B/J_0
double cost_J, cost_K;

double J_p = 1.0;                   // spin-spin coupling constant
vector < vector<int> > S( L, vector<int>(L,0) );  
vector < vector<int> > S_data( L, vector<int>(L,0) );  
vector < vector<int> > h( L, vector<int>(L,0) );  

vector < vector<double> > K_datax( L, vector<double>(L,0) ); 
vector < vector<double> > K_datay( L, vector<double>(L,0) ); 
vector < vector<double> > K_modlx( L, vector<double>(L,0) );
vector < vector<double> > K_modly( L, vector<double>(L,0) );

// J_modl[u][], x-direction: J_modl= i-to-j, => L*i+j, y-direction: J_modl= i-to-j, => L*i+j+L,
vector < vector<double> > J_modlx( L, vector<double>(L,0) ); 
vector < vector<double> > J_modly( L, vector<double>(L,0) ); 
vector < vector<double> > test_Jx( L, vector<double>(L,1.0) ); 
vector < vector<double> > test_Jy( L, vector<double>(L,1.0) ); 

template <typename T>
string to_string_with_precision(const T a_value, const int n = 5)
{
    ostringstream out;
    out << setprecision(n) << a_value;
    return out.str();
}

void init_S(){
	for(int i=0;i<L;++i){
		for(int j=0;j<L;++j){
		S[i][j] = int( 2*(rand()%2) - 1 );
	}}
}

void init_J_model(){
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){
		J_modlx[x][y] = 0.0;     
		J_modly[x][y] = 0.0;
	}}
}

bool read_file(string filename){
  bool flag = true;
  ifstream myfile (filename);
  string line;
  int count = 0; int i=0,j=0; // this initialize is necessary.
  if (myfile.is_open())
  {
    while ( getline (myfile,line,' ') && (i<L-1||j<L-1))
    {
		i = int(double(count)/L);	
		j = (count%L);	
		S_data[i][j] = stoi(line);
    count += 1;
    }
    myfile.close();
    flag = true;
  }
  else{ cout << "Unable to open file" << endl;  
  	cout << filename << endl;
  }
	return flag;
}

void set_test_J(){
	string fname = "./J_test-L"+to_string(L)+"-T"+to_string_with_precision(T)+"-MCMC.dat";
	ifstream myfile(fname);
	string line;
	int count = 0; int i=0; int j=0; // this initialize is necessary.
	cout<< "belows are test_Jx and test_Jy." << endl;
	double a;
	if (myfile.is_open())
	{
		//#while ( getline (myfile,line,' ') )
	    while ( getline (myfile,line) )
	    {	
		    myfile >> a;
			
		if(count < L*L){
			i = int(double(count)/L);	
			j = (count%L);	
			test_Jx[i][j] = a;	
		//	cout<<test_Jx[i][j]<<endl;
		}
		if( (count<2*L*L) && (L*L<count)  ){
			i = int(double(count)/L) - L;	
			j = (count%L);	
			test_Jy[i][j] = a;	
		//	cout<<test_Jy[i][j]<<endl;
		}
			count += 1;
	    }
	    myfile.close();
	}
	else{ cout << "Unable to open file" << endl;  
	}
}

double E_gene(int x, int y)
{
    int x_plus = (x+1)%L, x_minus = (x-1+L)%L, y_plus =(y+1)%L , y_minus = (y-1+L)%L;
    
    // Suppose short range interaction., Use J_modl not test_J. 
    double E = - S[x][y] * (
         J_modlx[x][y]*S[x_plus][y] + J_modlx[x_minus][y]*S[x_minus][y]
       + J_modly[x][y]*S[x][y_plus] + J_modly[x][y_minus]*S[x][y_minus] );
    return E;
}

double E()
{
    // total energy
    double E_sum = 0;
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y)
            E_sum += E_gene(x, y);
    return E_sum / 2;
}
//  m := q/(q-1) <(S_i -1/q)>, this formulation is correspond to the case of Ising(q=2).
double M(){
	double m=0;
	for(int x=0;x<L;++x){
		for(int y=0;y<L;++y){
		//m += (S[x][y] - 1.0/n_q);	
		m += S[x][y];	
	}}
	return m ;
}

bool metropolis_step(           // true/false if step accepted/rejected
    int x, int y,               // change spin at size x,y
    double delta = 1.0 )	// maximum step size in radians 
{
    // find the local energy at site (x,y)
    double E_xy = E_gene(x, y);

    // save the value of spin at (x,y)
    double S_save = S[x][y];

    S[x][y] *= -1;

    double E_xy_trial = E_gene(x, y);

    // Metropolis test
    bool accepted = false;
    double dE = E_xy_trial - E_xy;
    double w = exp(- dE / T);
    if (w > std_rand())
        accepted = true;
    else
        S[x][y] = S_save;
    return accepted;
}

// this monte_carlo_sweep shold use J_modl. 
int monte_carlo_sweep()
{
    int acceptances = 0;
    // randomized sweep
    for (int step = 0; step < N; ++step) {
        // choose a site at random
        int x = int(L * std_rand());
        int y = int(L * std_rand());
        if (metropolis_step(x, y))
            ++acceptances;
    }
    return acceptances;
}

void calc_K_data(){
	int x_plus,y_plus;
	int S_00,S_p0,S_0p,S_m0,S_0m;
	for(int n=0;n<n_sample;++n){
		string filename="./DataL10T"+to_string_with_precision(T)+"/state-L"+to_string(L)+"-T"+to_string_with_precision(T)+"-MCMC"+to_string(n*200)+".dat";
		//cout<<"filename="<<filename<<endl;
		read_file(filename);
		for(int x=0;x<L;++x){	
		for(int y=0;y<L;++y){
			x_plus=(x+1)%L,y_plus=(y+1)%L; 
			
			K_datax[x][y] += S_data[x][y]*S_data[x_plus][y]; 	
			K_datay[x][y] += S_data[x][y]*S_data[x][y_plus]; 	
		}}	
	}// End of sample loop.	

	// Normalization.
	double percent = 1.0/double(n_sample);
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){
		K_datax[x][y] *= percent;  
		K_datay[x][y] *= percent; 
	} }	
}

void inti_K_modl(){
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){
		K_modlx[x][y] = 0;	
		K_modly[x][y] = 0;	
	}}
}

void calc_K_model(){
	//int col_i,row_i,col_j,row_j,a,b;
	int x_plus,y_plus;
	int S_00,S_p0,S_0p;
    	int equilibration_steps = 50000;
    	int production_steps = 20000;
    	int np = 200;
	inti_K_modl();
	init_S();
	for (int step = 0; step < equilibration_steps; ++step){
		monte_carlo_sweep(); // this monte_carlo_sweep should use J_modl
	}
	for (int step = 0; step < production_steps; ++step){
		monte_carlo_sweep(); // this monte_carlo_sweep should use J_modl
		if(step%np==0){
			for(int x=0;x<L;++x){
			for(int y=0;y<L;++y){
				x_plus=(x+1)%L; y_plus=(y+1)%L;
				
				K_modlx[x][y] += S[x][y]*S[x_plus][y];
				K_modly[x][y] += S[x][y]*S[x][y_plus];
		}}	
		}
	}
	
	// Normalization.
	double percent = double(np)/double(production_steps);
		for(int x=0;x<L;++x){
		for(int y=0;y<L;++y){
			K_modlx[x][y] *= percent; 	
			K_modly[x][y] *= percent; 	
	} }	
}

void update_J_modl(){
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){

		J_modlx[x][y] +=  lr*( K_datax[x][y] - K_modlx[x][y] ); 
		//J_modlx[x][y] += - regula*J_modlx[x][y];

		J_modly[x][y] +=  lr*( K_datay[x][y] - K_modly[x][y] ); 
		//J_modly[x][y] += - regula*J_modly[x][y];

	}}
}

double evaluater_of_error(int epoc){
	cost_J=0; cost_K=0; 
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){

		cost_J+=abs(J_modlx[x][y] - test_Jx[x][y]);	
		cost_J+=abs(J_modly[x][y] - test_Jy[x][y]);	

		cost_K+=abs(K_datax[x][y] - K_modlx[x][y]);	
		cost_K+=abs(K_datay[x][y] - K_modly[x][y]);	
	}}
	cost_J /= double(2*N);
	cost_K /= double(2*N);
	if(epoc%100==0){
	string fname = "error"+to_string(int(double(epoc)/100))+".dat";
	ofstream file(fname);
	//print interaction of x direction//
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){
			// T* is correction.
		file << J_modlx[x][y] << " " << test_Jx[x][y] <<  endl;
		//	<< K_modlx[x][y] << " " <<K_datax[x][y] <<endl;
		}} 
	
	//print interaction of y direction//
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){
		file << J_modly[x][y] << " " << test_Jy[x][y] <<endl;
		//	<< K_modly[x][y] << " " <<K_datay[x][y] <<endl;
		}} 
	file.close();
	}
	return cost_J;
}

int main()
{
	set_test_J();
	calc_K_data();
	init_J_model();
	for(int epoc=0;epoc<epoc_stop;++epoc){
		/*calclation of the data.*/
		cout << epoc<<" " << evaluater_of_error(epoc) << " " << cost_K << endl;
		calc_K_model();	
		update_J_modl();
		}
        return 0;
}
