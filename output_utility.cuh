#include <set>
#include <fstream>
#include <string>
#include <sstream>

class OutputCollector {
  std::set<DATAPOINT> dp_set;
public:
  void add_datapoint(DATAPOINT& datapoint) {
    dp_set.insert(datapoint);
  }
  void saveCsvs(std::string perf_idx_filename,std::string time_idx_filename) {
    std::ofstream csvfile;
    csvfile.open(perf_idx_filename);
    std::set<DATAPOINT>::iterator it;
    for (it=dp_set.begin(); it!=std::prev(dp_set.end()); it++)
	    csvfile << it->perf_idx << ",";
    csvfile << it->perf_idx << "\n";
    csvfile.close();

    csvfile.open(time_idx_filename);
    for (it=dp_set.begin(); it!=std::prev(dp_set.end()); it++)
	    csvfile << it->time_idx << ",";
    csvfile << it->time_idx << "\n";
    csvfile.close();
  }
};

class NormErrorCollector {
  std::ostringstream dump_stream;
public:
  template<int n_values>
  void add_values(float *values) {
    for(int i=0; i<n_values; i++) {
      dump_stream << values[i] << ",";
    }
  }
  void saveCsv(std::string filename) {
    dump_stream.seekp(-1, dump_stream.cur);
    dump_stream << "\n";
    std::ofstream csvfile;
    csvfile.open(filename);
    csvfile << dump_stream.str();
    csvfile.close();
  }
};
