// This file has been deprecated and is preserved for backwards
// compatability only. Catapult will use the latest versions of
// the components automatically for new designs

#ifndef INCLUDED_MGC_IOPORT_TRANS_RSC_H
#define INCLUDED_MGC_IOPORT_TRANS_RSC_H

#include "mc_transactors.h"

////////////////////////////////////////////////////////////////////////////////
// mgc_in_wire_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
// This is a purely combinational interface component
//
template<int streamcnt, int width>
class mgc_in_wire_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::port_type port_type;
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       z;

  // Next three functions are overrides for combinational behavior
  virtual bool is_combinational() { return true; }
  virtual void require_driving_value_adjustments(int RH, int BaseAddr, int CH, int BaseBit) {
    const int row = this->get_current_in_row();
    if (BaseAddr <= row && row <= RH) {
      this->_value_changed.notify(SC_ZERO_TIME);
    }
  }
  virtual void adjust_driving_value(int row, int idx_lhs, int vwidth, sc_lv_base& rhs, int rhs_idx) {
    if (row == this->get_current_in_row()) {
      this->set_value(DRV, idx_lhs, vwidth, rhs, rhs_idx);
    }
  }

  SC_HAS_PROCESS(mgc_in_wire_trans_rsc_v1);
  mgc_in_wire_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0)
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    MC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  // a simple wire without handshake cannot be used for streamed data. force row to always be zero
  virtual int get_current_in_row() const { return 0; }

  void clk_skew_delay() {
    this->exchange_value();
  }

  void update_z() {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      z = this->read_row(DRV);
  }

  void my_at_active_clk() { base::at_active_clk(); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_in_wire_en_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_in_wire_en_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       z;
  sc_in<sc_dt::sc_logic>  lz;

  SC_HAS_PROCESS(mgc_in_wire_en_trans_rsc_v1);
  mgc_in_wire_en_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    if (lz.read() == SC_LOGIC_1) {
      this->incr_current_in_row(); // value is being read right now, so advance index
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value();
  }

  void update_z() {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      z = this->read_row(DRV);
  }

  void my_at_active_clk() { base::at_active_clk(); }
};


////////////////////////////////////////////////////////////////////////////////
// mgc_in_wire_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_in_wire_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_out<sc_dt::sc_logic> vz;

  SC_HAS_PROCESS(mgc_in_wire_wait_trans_rsc_v1);
  mgc_in_wire_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , vz("vz")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(0/*input*/) && io_request(0/*input*/);
    if ((lz.read() == SC_LOGIC_1) && !waiting) {
      this->incr_current_in_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      z = this->read_row(DRV);
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void my_wait_controller() { base::wait_controller(0/*input*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir) { return (lz.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_chan_in_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width, int sz_width>
class mgc_chan_in_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_out<sc_dt::sc_logic> vz;
  sc_out<sc_lv<sz_width> > sz;
  sc_in<sc_dt::sc_logic>  slz;

  SC_HAS_PROCESS(mgc_chan_in_trans_rsc_v1);
  mgc_chan_in_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , vz("vz")
    , sz("sz")
    , slz("slz")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];

    SC_METHOD(slz_event);
    this->sensitive << this->_update_sz;
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  virtual void start_of_simulation() {
    sz.write(0); // Drive initial value on size port to avoid X's
  }

  void slz_event() {
    int tmp = this->count_sizelz_event(slz.read()==SC_LOGIC_1); // record fact that H/W attempted sz read
    if (slz.read() != SC_LOGIC_0) {
       sz.write(this->in_used()); // update sz with current fifo used count from transactor
    }
  }

  void clk_skew_delay() {
    this->_update_sz.notify(SC_ZERO_TIME);
    bool waiting = this->wait_signal_active(0/*input*/) && io_request(0/*input*/);
    if ((lz.read() == SC_LOGIC_1) && !waiting) {
      this->incr_current_in_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
     this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
      sz.write(this->in_used()); // update sz with current fifo used count from transactor
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      z = this->read_row(DRV);
  }
  sc_event _update_sz; // need to update sz (if slz hi)

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_wait_controller()                 { base::wait_controller(0/*input*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (lz.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_out_stdreg_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_out_stdreg_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;

  SC_HAS_PROCESS(mgc_out_stdreg_trans_rsc_v1);
  mgc_out_stdreg_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  // a simple wire without handshake cannot be used for streamed data. force row to always be zero
  virtual int get_current_in_row() const { return 0; }

  void clk_skew_delay() {
    this->write_row(0, this->z.read());
    this->exchange_value();
  }

  void my_at_active_clk() { base::at_active_clk(); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_out_stdreg_en_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_out_stdreg_en_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;

  SC_HAS_PROCESS(mgc_out_stdreg_en_trans_rsc_v1);
  mgc_out_stdreg_en_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    if (this->lz.read() == SC_LOGIC_1) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    this->exchange_value();
  }

  void my_at_active_clk() { base::at_active_clk(); }
};


////////////////////////////////////////////////////////////////////////////////
// mgc_out_stdreg_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_out_stdreg_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_out<sc_dt::sc_logic> vz;

  SC_HAS_PROCESS(mgc_out_stdreg_wait_trans_rsc_v1);
  mgc_out_stdreg_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , vz("vz")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lz.read() == SC_LOGIC_1) && !waiting) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    // always prepare for exchange (should triosy be true) unless we are waiting AND
    // output is currently being written.
    this->exchange_value(waiting & (this->lz.read() == SC_LOGIC_1));
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_wait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (lz.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_out_reg_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width, int ph_clk, int ph_en, int ph_arst, int ph_srst>
class mgc_out_reg_wait_trans_rsc_v1
  : public mgc_out_stdreg_wait_trans_rsc_v1<streamcnt,width>
{
public:
  sc_in<sc_dt::sc_logic> en;
  sc_in<sc_dt::sc_logic> arst; // unused by transactor model
  sc_in<sc_dt::sc_logic> srst; // unused by transactor model

  SC_HAS_PROCESS(mgc_out_reg_wait_trans_rsc_v1);
  mgc_out_reg_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : mgc_out_stdreg_wait_trans_rsc_v1<streamcnt,width>(name, phase, clk_skew_delay)
    , en("en")
    , arst("arst")
    , srst("srst")
    {}

};

////////////////////////////////////////////////////////////////////////////////
// mgc_out_reg_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int rscid, int width, int ph_clk, int ph_en, int ph_arst, int ph_srst>
class mgc_out_reg_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_in<sc_dt::sc_logic>  arst; // unused in this model
  sc_in<sc_dt::sc_logic>  srst; // unused in this model
  sc_in<sc_dt::sc_logic>  en; // unused in this model

  SC_HAS_PROCESS(mgc_out_reg_trans_rsc_v1);
  mgc_out_reg_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , arst("arst")
    , srst("srst")
    , en("en")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    if (this->lz.read() == SC_LOGIC_1) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    this->exchange_value();
  }

  void my_at_active_clk() { base::at_active_clk(); }
};


////////////////////////////////////////////////////////////////////////////////
// mgc_out_prereg_en_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////////////
// mc_rsc_registered_wire
//    Derived resource class suitable for wires with registered inputs.
//
// This resource block has (logically) registers for all inputs. So it
// does not need any combinational process at all. If at clock edge
// both, lzin and triosy are high (ie. the call to exchange_value does
// modify the contents), then we need to show the NEW value, since the
// correponding triosync operation happened at the previous clock
// edge, but the corresponding read operation happens at this current
// edge, thus AFTER the tiosync operation. If lzout and triosy are
// high, then we also write to the NEW value for the very same
// reasons. Thus exchange_value(...) needs to be called before we
// react to the port values.
//
// For channels, which don't rely on triosy, the capture occurs when
// lz is active. This means the exchange_value() call must not be early.
// Use _capture_data_reg to control when exchange_value() is called.
////////////////////////////////////////////////////////////////////////////////
template<int streamcnt, int width>
class mgc_out_prereg_en_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;
  const bool _capture_data_reg;

  SC_HAS_PROCESS(mgc_out_prereg_en_trans_rsc_v1);
  mgc_out_prereg_en_trans_rsc_v1(const sc_module_name& name, bool phase, bool capture_data_reg=true, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , _capture_data_reg(capture_data_reg)
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  virtual void inject_value(int row, int idx_lhs, int vwidth, sc_lv_base& rhs, int idx_rhs) {
    this->set_value(row, idx_lhs, vwidth, rhs, idx_rhs);
  }

  virtual void extract_value(int row, int idx_rhs, int vwidth, sc_lv_base& lhs, int idx_lhs) {
    this->get_value(row, idx_rhs, vwidth, lhs, idx_lhs);
  }

  void clk_skew_delay() {
    if (this->_capture_data_reg)
        this->exchange_value();
    if (this->lz.read() == SC_LOGIC_1) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    if (!this->_capture_data_reg)
        this->exchange_value();
  }

  void my_at_active_clk() { base::at_active_clk(); }
};


////////////////////////////////////////////////////////////////////////////////
// mgc_out_buf_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int rscid, int width, int ph_clk, int ph_en, int ph_arst, int ph_srst>
class mgc_out_buf_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_out<sc_dt::sc_logic> vz;
  sc_in<sc_dt::sc_logic>  arst; // unused in this model
  sc_in<sc_dt::sc_logic>  srst; // unused in this model
  sc_in<sc_dt::sc_logic>  en; // unused in this model

  SC_HAS_PROCESS(mgc_out_buf_wait_trans_rsc_v1);
  mgc_out_buf_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , vz("vz")
    , arst("arst")
    , srst("srst")
    , en("en")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lz.read() == SC_LOGIC_1) && !waiting) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    // always prepare for exchange (should triosy be true) unless we are waiting AND
    // output is currently being written.
    this->exchange_value(waiting & (this->lz.read() == SC_LOGIC_1));
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_wait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (lz.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_out_fifo_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int rscid, int width, int fifo_sz, int ph_clk, int ph_en, int ph_arst, int ph_srst>
class mgc_out_fifo_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_out<sc_dt::sc_logic> vz;
  sc_in<sc_dt::sc_logic>  arst; // unused in this model
  sc_in<sc_dt::sc_logic>  srst; // unused in this model
  sc_in<sc_dt::sc_logic>  en; // unused in this model

  SC_HAS_PROCESS(mgc_out_fifo_wait_trans_rsc_v1);
  mgc_out_fifo_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , vz("vz")
    , arst("arst")
    , srst("srst")
    , en("en")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lz.read() == SC_LOGIC_1) && !waiting) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    // always prepare for exchange (should triosy be true) unless we are waiting AND
    // output is currently being written.
    this->exchange_value(waiting & (this->lz.read() == SC_LOGIC_1));
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_wait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (lz.read() == SC_LOGIC_1); }
};



////////////////////////////////////////////////////////////////////////////////
// mgc_out_decouple_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int rscid, int width, int ph_clk, int ph_en, int ph_arst, int ph_srst>
class mgc_out_decouple_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<data_type>        z;
  sc_in<sc_dt::sc_logic>  lz;
  sc_out<sc_dt::sc_logic> vz;
  sc_in<sc_dt::sc_logic>  arst; // unused in this model
  sc_in<sc_dt::sc_logic>  srst; // unused in this model
  sc_in<sc_dt::sc_logic>  en; // unused in this model

  SC_HAS_PROCESS(mgc_out_decouple_trans_rsc_v1);
  mgc_out_decouple_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , lz("lz")
    , vz("vz")
    , arst("arst")
    , srst("srst")
    , en("en")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lz.read() == SC_LOGIC_1) && !waiting) {
      this->write_row(this->get_current_out_row(), this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    // always prepare for exchange (should triosy be true) unless we are waiting AND
    // output is currently being written.
    this->exchange_value(waiting & (this->lz.read() == SC_LOGIC_1));
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_wait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (lz.read() == SC_LOGIC_1); }
};


////////////////////////////////////////////////////////////////////////////////
// mgc_inout_prereg_en_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_inout_prereg_en_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       zin;
  sc_in<data_type>        zout;
  sc_in<sc_dt::sc_logic>  lzout;

  // Next three functions are overrides for combinational behavior
  virtual bool is_combinational() { return true; }
  virtual void require_driving_value_adjustments(int RH, int BaseAddr, int CH, int BaseBit) {
    const int row = this->get_current_in_row();
    if (BaseAddr <= row && row <= RH) {
      this->_value_changed.notify(SC_ZERO_TIME);
    }
  }
  virtual void adjust_driving_value(int row, int idx_lhs, int vwidth, sc_lv_base& rhs, int rhs_idx) {
    if (row == this->get_current_in_row()) {
      this->set_value(DRV, idx_lhs, vwidth, rhs, rhs_idx);
    }
  }

  SC_HAS_PROCESS(mgc_inout_prereg_en_trans_rsc_v1);
  mgc_inout_prereg_en_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , zin("zin")
    , zout("zout")
    , lzout("lzout")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed << zout << lzout;
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    this->exchange_value();
    if (this->lzout.read() == SC_LOGIC_1) {
      this->write_row(this->get_current_out_row(), this->zout.read());
      this->incr_current_out_row();
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
  }

  void update_z() {
    if (1) {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      zin = this->read_row(DRV);
    }
  }

  void my_at_active_clk() { base::at_active_clk(); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_inout_stdreg_en_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_inout_stdreg_en_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       zin;
  sc_in<data_type>        zout;
  sc_in<sc_dt::sc_logic>  lzin;
  sc_in<sc_dt::sc_logic>  lzout;

  SC_HAS_PROCESS(mgc_inout_stdreg_en_trans_rsc_v1);
  mgc_inout_stdreg_en_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , zin("zin")
    , zout("zout")
    , lzin("lzin")
    , lzout("lzout")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << lzin << this->_value_changed;
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    if (this->lzout.read() == SC_LOGIC_1) {
      this->write_row(this->get_current_out_row(), this->zout.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    } else if (this->lzin.read() == SC_LOGIC_1) {
      this->incr_current_in_row(); // value is being read right now, so advance index
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value();
  }

  void update_z() {
    if (lzin.read() == SC_LOGIC_1) {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      zin = this->read_row(DRV);
    }
  }

  void my_at_active_clk() { base::at_active_clk(); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_inout_stdreg_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class mgc_inout_stdreg_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       zin;
  sc_in<data_type>        zout;
  sc_in<sc_dt::sc_logic>  lzin;
  sc_in<sc_dt::sc_logic>  lzout;
  sc_out<sc_dt::sc_logic> vzin;
  sc_out<sc_dt::sc_logic> vzout;

  SC_HAS_PROCESS(mgc_inout_stdreg_wait_trans_rsc_v1);
  mgc_inout_stdreg_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , zin("zin")
    , zout("zout")
    , lzin("lzin")
    , lzout("lzout")
    , vzin("vzin")
    , vzout("vzout")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << lzin << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_INwait_controller);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];
    this->dont_initialize();

    SC_METHOD(my_OUTwait_controller);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lzout.read() == SC_LOGIC_1) && !this->wait_signal_active(1/*output*/)) {
      this->write_row(this->get_current_out_row(), this->zout.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    } else if ((this->lzin.read() == SC_LOGIC_1) && !this->wait_signal_active(0/*input*/)) {
      this->incr_current_in_row(); // value is being read right now, so advance index
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
    if (lzin.read() == SC_LOGIC_1) {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      zin = this->read_row(DRV);
    } else {
//       const static data_type zv('Z');
//       z = zv;
    }
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_INwait_controller()                 { base::wait_controller(0/*input*/); }
  void         my_OUTwait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { if (!iodir) {vzin = wait_active ? SC_LOGIC_0 : SC_LOGIC_1;} else {vzout = wait_active ? SC_LOGIC_0 : SC_LOGIC_1;}}

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (iodir==0) ? (vzin.read() == SC_LOGIC_0) : (vzout.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (iodir==0) ? (lzin.read() == SC_LOGIC_1) : (lzout.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_inout_buf_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int rscid, int width, int ph_clk, int ph_en, int ph_arst, int ph_srst>
class mgc_inout_buf_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       zin;
  sc_in<data_type>        zout;
  sc_in<sc_dt::sc_logic>  lzin;
  sc_in<sc_dt::sc_logic>  lzout;
  sc_out<sc_dt::sc_logic> vzin;
  sc_out<sc_dt::sc_logic> vzout;
  sc_in<sc_dt::sc_logic>  arst; // unused in this model
  sc_in<sc_dt::sc_logic>  srst; // unused in this model
  sc_in<sc_dt::sc_logic>  en; // unused in this model

  SC_HAS_PROCESS(mgc_inout_buf_wait_trans_rsc_v1);
  mgc_inout_buf_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , zin("zin")
    , zout("zout")
    , lzin("lzin")
    , lzout("lzout")
    , vzin("vzin")
    , vzout("vzout")
    , en("en")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << lzin << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_INwait_controller);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];
    this->dont_initialize();

    SC_METHOD(my_OUTwait_controller);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lzout.read() == SC_LOGIC_1) && !this->wait_signal_active(1/*output*/)) {
      this->write_row(this->get_current_out_row(), this->zout.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    } else if ((this->lzin.read() == SC_LOGIC_1) && !this->wait_signal_active(0/*input*/)) {
      this->incr_current_in_row(); // value is being read right now, so advance index
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
    if (lzin.read() == SC_LOGIC_1) {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      zin = this->read_row(DRV);
    } else {
//       const static data_type zv('Z');
//       z = zv;
    }
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_INwait_controller()                 { base::wait_controller(0/*input*/); }
  void         my_OUTwait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { if (!iodir) {vzin = wait_active ? SC_LOGIC_0 : SC_LOGIC_1;} else {vzout = wait_active ? SC_LOGIC_0 : SC_LOGIC_1;}}

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (iodir==0) ? (vzin.read() == SC_LOGIC_0) : (vzout.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (iodir==0) ? (lzin.read() == SC_LOGIC_1) : (lzout.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// mgc_inout_fifo_wait_trans_rsc_v1
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int rscid, int width, int fifo_sz, int ph_clk, int ph_en, int ph_arst, int ph_srst, int ph_log2>
class mgc_inout_fifo_wait_trans_rsc_v1
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       zin;
  sc_in<data_type>        zout;
  sc_in<sc_dt::sc_logic>  lzin;
  sc_in<sc_dt::sc_logic>  lzout;
  sc_out<sc_dt::sc_logic> vzin;
  sc_out<sc_dt::sc_logic> vzout;
  sc_in<sc_dt::sc_logic>  arst; // unused in this model
  sc_in<sc_dt::sc_logic>  srst; // unused in this model
  sc_in<sc_dt::sc_logic>  en; // unused in this model

  SC_HAS_PROCESS(mgc_inout_fifo_wait_trans_rsc_v1);
  mgc_inout_fifo_wait_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , zin("zin")
    , zout("zout")
    , lzin("lzin")
    , lzout("lzout")
    , vzin("vzin")
    , vzout("vzout")
    , en("en")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << lzin << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_INwait_controller);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];
    this->dont_initialize();

    SC_METHOD(my_OUTwait_controller);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lzout.read() == SC_LOGIC_1) && !this->wait_signal_active(1/*output*/)) {
      this->write_row(this->get_current_out_row(), this->zout.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    } else if ((this->lzin.read() == SC_LOGIC_1) && !this->wait_signal_active(0/*input*/)) {
      this->incr_current_in_row(); // value is being read right now, so advance index
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
    if (lzin.read() == SC_LOGIC_1) {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      zin = this->read_row(DRV);
    } else {
//       const static data_type zv('Z');
//       z = zv;
    }
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_INwait_controller()                 { base::wait_controller(0/*input*/); }
  void         my_OUTwait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { if (!iodir) {vzin = wait_active ? SC_LOGIC_0 : SC_LOGIC_1;} else {vzout = wait_active ? SC_LOGIC_0 : SC_LOGIC_1;}}

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (iodir==0) ? (vzin.read() == SC_LOGIC_0) : (vzout.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (iodir==0) ? (lzin.read() == SC_LOGIC_1) : (lzout.read() == SC_LOGIC_1); }
};


////////////////////////////////////////////////////////////////////////////////
// modulario_en_in from modular_io.h
//    
////////////////////////////////////////////////////////////////////////////////
//
//
template<int streamcnt, int width>
class modulario_en_in_trans_rsc
  : public mc_wire_trans_rsc_base<width,streamcnt>
{
public:
  typedef mc_wire_trans_rsc_base<width,streamcnt> base;
  MC_EXPOSE_NAMES_OF_BASE(base);
  typedef typename base::data_type data_type;

  enum { COLS = base::COLS };
  enum { DRV = streamcnt}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_out<data_type>       z;
  sc_out<sc_dt::sc_logic> vz;

  SC_HAS_PROCESS(modulario_en_in_trans_rsc);
  modulario_en_in_trans_rsc(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , z("z")
    , vz("vz")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(0/*input*/) && io_request(0/*input*/);
    if ( !waiting) {
      this->incr_current_in_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
      const int row = this->get_current_in_row();
      this->write_row(DRV, this->read_row(row));
      if (this->is_combinational()) 
        this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
      z = this->read_row(DRV);
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void my_wait_controller() { base::wait_controller(0/*input*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vz' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vz = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vz.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lz' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir) { return false;}
};


////////////////////////////////////////////////////////////////////////////////
// ccs_event_in_trans_rsc_v1
////////////////////////////////////////////////////////////////////////////////
class ccs_event_in_trans_rsc_v1 : public mc_wire_trans_rsc_base<1,1>
{
public:
  typedef mc_wire_trans_rsc_base<1,1> base;
  MC_EXPOSE_NAMES_OF_BASE(base);

  enum { COLS = base::COLS };
  enum { DRV = 1 }; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<sc_dt::sc_logic>  lzin;

  SC_HAS_PROCESS(ccs_event_in_trans_rsc_v1);
  ccs_event_in_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , lzin("lzin")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    if (lzin.read() == SC_LOGIC_1) {
      this->incr_current_in_row(); // value is being read right now, so advance index
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value();
  }

  void update_z() {
    const int row = this->get_current_in_row();
    this->write_row(DRV, this->read_row(row));
    if (this->is_combinational()) 
      this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
    /*z = */ this->read_row(DRV);
  }

  void my_at_active_clk() { base::at_active_clk(); }
};

////////////////////////////////////////////////////////////////////////////////
// ccs_sync_in_trans_rsc_v1
////////////////////////////////////////////////////////////////////////////////
class ccs_sync_in_trans_rsc_v1 : public mc_wire_trans_rsc_base<1,1>
{
public:
  typedef mc_wire_trans_rsc_base<1,1> base;
  MC_EXPOSE_NAMES_OF_BASE(base);

  enum { COLS = base::COLS };
  enum { DRV = 1 }; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<sc_dt::sc_logic>  lzin;
  sc_out<sc_dt::sc_logic> vzin;

  SC_HAS_PROCESS(ccs_sync_in_trans_rsc_v1);
  ccs_sync_in_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , lzin("lzin")
    , vzin("vzin")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(update_z);
    this->sensitive << this->_value_changed;
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[0/*input*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(0/*input*/) && io_request(0/*input*/);
    if ((lzin.read() == SC_LOGIC_1) && !waiting) {
      this->incr_current_in_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - input index now " << this->get_current_in_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
      this->_value_changed.notify(SC_ZERO_TIME); // display next row
    }
    this->exchange_value(waiting);
  }

  void update_z() {
    const int row = this->get_current_in_row();
    this->write_row(DRV, this->read_row(row));
    if (this->is_combinational()) 
      this->initiate_driving_value_adjustments(row, row, COLS - 1, 0);
    vzin.write(this->read_row(DRV)=="1"?SC_LOGIC_1:SC_LOGIC_0); // this needs to drive vzin !!!
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void my_wait_controller() { base::wait_controller(0/*input*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vzin' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vzin = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vzin.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lzin' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir) { return (lzin.read() == SC_LOGIC_1); }
};

////////////////////////////////////////////////////////////////////////////////
// ccs_event_out_trans_rsc_v1 
////////////////////////////////////////////////////////////////////////////////
class ccs_event_out_trans_rsc_v1 : public mc_wire_trans_rsc_base<1,1>
{
public:
  typedef mc_wire_trans_rsc_base<1,1> base;
  MC_EXPOSE_NAMES_OF_BASE(base);

  enum { COLS = base::COLS };
  enum { DRV = 1}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<sc_dt::sc_logic>  lzout;

  SC_HAS_PROCESS(ccs_event_out_trans_rsc_v1);
  ccs_event_out_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , lzout("lzout")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    if (this->lzout.read() == SC_LOGIC_1) {
      this->write_row(this->get_current_out_row(), sc_lv<1>("1"));//this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    this->exchange_value();
  }

  void my_at_active_clk() { base::at_active_clk(); }
};

////////////////////////////////////////////////////////////////////////////////
// ccs_sync_out_trans_rsc_v1 
////////////////////////////////////////////////////////////////////////////////
class ccs_sync_out_trans_rsc_v1 : public mc_wire_trans_rsc_base<1,1>
{
public:
  typedef mc_wire_trans_rsc_base<1,1> base;
  MC_EXPOSE_NAMES_OF_BASE(base);

  enum { COLS = base::COLS };
  enum { DRV = 1}; // driving value is stored at row position streamcnt of data

  sc_in<bool>             clk;
  sc_in<sc_dt::sc_logic>  lzout;
  sc_out<sc_dt::sc_logic> vzout;

  SC_HAS_PROCESS(ccs_sync_out_trans_rsc_v1);
  ccs_sync_out_trans_rsc_v1(const sc_module_name& name, bool phase, double clk_skew_delay=0.0) 
    : base(name,phase,clk_skew_delay)
    , clk("clk")
    , lzout("lzout")
    , vzout("vzout")
  {
    MC_METHOD(my_at_active_clk);
    this->sensitive << (phase ? this->clk.pos() : this->clk.neg()); //active edge
    this->dont_initialize();

    SC_METHOD(my_wait_controller);
    this->sensitive << (phase ? this->clk.neg() : this->clk.pos()); // wait control restored on inactive clk edge
    this->sensitive << this->_wait_cycles_changed[1/*output*/];

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  void clk_skew_delay() {
    bool waiting = this->wait_signal_active(1/*output*/) && io_request(1/*output*/);
    if ((this->lzout.read() == SC_LOGIC_1) && !waiting) {
      this->write_row(this->get_current_out_row(), sc_lv<1>("1"));//this->z.read());
      this->incr_current_out_row();
      if (this->is_stream() && this->log_event(MC_TRANSACTOR_STREAMCNT)) {
        std::ostringstream msg;
        msg << "MC_TRANSACTOR_STREAMCNT - output index now " << this->get_current_out_row() << " @ " << sc_time_stamp();
        SC_REPORT_INFO(this->name(), msg.str().c_str());
      }
    }
    // always prepare for exchange (should triosy be true) unless we are waiting AND
    // output is currently being written.
    this->exchange_value(waiting & (this->lzout.read() == SC_LOGIC_1));
  }

  void my_at_active_clk() { base::at_active_clk(); }

  // Wait controller
  //   Define function sensitive to this transactor resource's clock and the _wait_cycles_changed event
  //   (this function must always only call the wait_controller() in the base transactor resource class).
  void         my_wait_controller()                 { base::wait_controller(1/*output*/); }

  //   Define function which drives this transactor resource's handshake signals high/low based on
  //   the boolean argument. If you copy this transactor resource class, change the 'vzout' to match the
  //   wait handshake signal of your interface component and adjust the polarity as needed.
  virtual void drive_wait_signals(bool wait_active, unsigned short iodir) { vzout = wait_active ? SC_LOGIC_0 : SC_LOGIC_1; }

  //   Define function which returns true if the wait handshake signal is active (waiting).
  virtual bool wait_signal_active(unsigned short iodir)                 { return (vzout.read() == SC_LOGIC_0); }

  //   Define function which returns true if the design is actively requesting data from this transactor resource.
  //   If you copy this transactor resource class, change the 'lzout' to match the data requested handshake
  //   signal of your interface component and adjust the polarity as needed.
  virtual bool io_request(unsigned short iodir)                    { return (lzout.read() == SC_LOGIC_1); }
};

#endif
