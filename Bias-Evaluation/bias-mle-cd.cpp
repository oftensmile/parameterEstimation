#include <cmath>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
void init();
vector<int> decimal_to_binary(int n);

struct Spin{
	int id;
	vector<int> state;     // spin variable at each lattice site
};
const int N = 8;                          // number of spins in x and y
double J_0 = 1.0;                   // spin-spin coupling constant
double f = 0.0;                     // uniform frustration constant
double T;                           // temperature in units of k_B/J_0

vector<double> h0;     // spin variable at each lattice site
vector<double> J0;       //
vector<double> h;     // spin variable at each lattice site
vector<double> J;       //
vector<Spin> spins;

void init(){
for(int i=0; i<N; ++i){
	Spin a = {i};
	//Spin a = {i,decimal_to_binary(i)};
	//Spin a = {i,hist(i)};
	spins.push_back(a);
	}
}

vector<int> decimal_to_binary(int n){
	string binary = bitset<N>(n).to_string(); //to binary
	vector<int> v(N);
	for(int j=0; j<N;++j){
		v[j] = binary[j];
	}
	return v;
}

double E(int n){
	
	vector<int> v=decimal_to_binary(n);
	

}

int main(int argc, char* argv[]){
	init();
	for(int n=0; n<N; ++n){
	cout<<spins[n].id<<"\n";
	}
	}
	

