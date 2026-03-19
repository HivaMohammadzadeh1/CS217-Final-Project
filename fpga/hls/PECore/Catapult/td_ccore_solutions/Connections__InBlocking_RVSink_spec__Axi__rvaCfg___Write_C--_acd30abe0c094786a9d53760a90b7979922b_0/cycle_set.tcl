
# Loop constraints
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp CSTEPS_FROM {{. == 0}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main CSTEPS_FROM {{. == 3} {.. == 0}}

# IO operation constraints
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/io_read(ccs_ccore_start:rsc.@) CSTEPS_FROM {{.. == 1}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/this.rdy.write:asn(this.rdy) CSTEPS_FROM {{.. == 1}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/this.rdy.write#1:asn(this.rdy) CSTEPS_FROM {{.. == 2}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/this.read_msg:asn(marshaller.Marshaller:v)#1 CSTEPS_FROM {{.. == 2}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/marshaller.AddField<ac_int<128,false>,128>:else:bits:io_write(data.data:rsc.@) CSTEPS_FROM {{.. == 2}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/marshaller.AddField<ac_int<24,false>,24>:else:bits:io_write(data.addr:rsc.@) CSTEPS_FROM {{.. == 2}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/marshaller.AddField<ac_int<1,false>,1>:else:bits:io_write(data.rw:rsc.@) CSTEPS_FROM {{.. == 2}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/return:asn CSTEPS_FROM {{.. == 2}}
directive set /Connections::InBlocking<RVSink<spec::Axi::rvaCfg>::Write,Connections::SYN_PORT>::PopNB/core/core:rlp/main/return:io_write(return:rsc.@) CSTEPS_FROM {{.. == 2}}

# Sync operation constraints

# Real operation constraints

# Probe constraints
