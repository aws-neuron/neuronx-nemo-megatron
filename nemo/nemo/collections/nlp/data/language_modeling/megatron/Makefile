CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell python3 -m pybind11 --includes)
LIBNAME = helpers
PYTHON_PATH = $(shell readlink -f $(shell which python3))
ifneq ("$(wildcard $(PYTHON_PATH)m-config)","")
	LIBEXT = $(shell $(PYTHON_PATH)m-config --extension-suffix)
else ifneq ("$(wildcard $(PYTHON_PATH)-config)","")
	LIBEXT = $(shell $(PYTHON_PATH)-config --extension-suffix)
else
	$(error "python3-config not found. Please run 'sudo apt install -y python3-dev' on Ubuntu or 'sudo yum install -y python3-devel' on Amazon Linux.")
endif

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
