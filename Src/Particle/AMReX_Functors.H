#ifndef AMREX_FUNCTORS_H_
#define AMREX_FUNCTORS_H_

#include <AMReX_Geometry.H>
#include <AMReX_BaseFab.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleUtil.H>

#ifdef AMREX_USE_CUDA

#include <thrust/tuple.h>


namespace amrex
{

namespace functors
{
  
    struct applyParticlePeriodicBCs
    {
        GeometryData domain;

        AMREX_GPU_HOST_DEVICE
        applyParticlePeriodicBCs(const GeometryData& a_domain)
            : domain(a_domain) {}

        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
        Particle operator()(const Particle& in) const {
            Particle p = in;
            
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
	      if (not domain.isPeriodic(dim)) continue;
	      if (p.m_rdata.pos[dim] > domain.ProbHi()[dim]) { 
		p.m_rdata.pos[dim] -= (domain.ProbHi()[dim]-domain.ProbLo()[dim]); 
	      } else if (p.m_rdata.pos[dim] < domain.ProbLo()[dim]) {
		p.m_rdata.pos[dim] += (domain.ProbHi()[dim]-domain.ProbLo()[dim]);
	      }
            }
            
            return p;
        }
    };

    struct assignParticleGrid
    {        
        GeometryData domain;
        Box box;
        BaseFab<int>* mask_ptr;
        
        AMREX_GPU_HOST_DEVICE
        assignParticleGrid(const GeometryData& a_domain,
                       const Box&          a_box,
                       BaseFab<int>*       a_mask_ptr) 
            : domain(a_domain), box(a_box), mask_ptr(a_mask_ptr) {}
        
        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
        int operator()(const Particle& p) const {
	    IntVect iv = IntVect(
	AMREX_D_DECL(floor((p.m_rdata.pos[0]-domain.ProbLo()[0])/domain.CellSize()[0]),
		     floor((p.m_rdata.pos[1]-domain.ProbLo()[1])/domain.CellSize()[1]),
		     floor((p.m_rdata.pos[2]-domain.ProbLo()[2])/domain.CellSize()[2]))
				 );

	    iv += domain.Domain().smallEnd();

	    return (*mask_ptr)(iv); 
        }
    };

    struct assignParticleTile
    {        
        GeometryData domain;
        Box box;
        BaseFab<int>* mask_ptr;
        
        AMREX_GPU_HOST_DEVICE
        assignParticleTile(const GeometryData& a_domain,
			   const Box&          a_box,
			   BaseFab<int>*       a_mask_ptr) 
            : domain(a_domain), box(a_box), mask_ptr(a_mask_ptr) {}
        
        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
        int operator()(const Particle& p) const {
	    IntVect iv = IntVect(
	AMREX_D_DECL(floor((p.m_rdata.pos[0]-domain.ProbLo()[0])/domain.CellSize()[0]),
		     floor((p.m_rdata.pos[1]-domain.ProbLo()[1])/domain.CellSize()[1]),
		     floor((p.m_rdata.pos[2]-domain.ProbLo()[2])/domain.CellSize()[2]))
				 );

	    iv += domain.Domain().smallEnd();

	    return (*mask_ptr)(iv, 1);
        }
    };

    struct assignParticleBin
    {
        GeometryData domain;
        Box box;
        IntVect bin_size;

        AMREX_GPU_HOST_DEVICE
        assignParticleBin(const GeometryData& a_domain,
			  const Box&          a_box,
			  const IntVect&      a_bin_size) 
            : domain(a_domain), box(a_box), bin_size(a_bin_size) {}
      
        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
	int operator()(const Particle& p) const {
	    IntVect iv = IntVect(
	AMREX_D_DECL(floor((p.m_rdata.pos[0]-domain.ProbLo()[0])/domain.CellSize()[0]),
		     floor((p.m_rdata.pos[1]-domain.ProbLo()[1])/domain.CellSize()[1]),
		     floor((p.m_rdata.pos[2]-domain.ProbLo()[2])/domain.CellSize()[2]))
				 );

	iv += domain.Domain().smallEnd();

        //
        // This function must be consistent with FabArrayBase::buildTileArray function!!!
        //
        auto tiling_1d = [](int i, int lo, int hi, int tilesize,
                            int& ntile, int& tileidx, int& tlo, int& thi) {
            int ncells = hi-lo+1;
            ntile = amrex::max(ncells/tilesize, 1);
            int ts_right = ncells/ntile;
            int ts_left  = ts_right+1;
            int nleft = ncells - ntile*ts_right;
	    int ii = i - lo;
            int nbndry = nleft*ts_left;
            if (ii < nbndry) {
                tileidx = ii / ts_left; // tiles on the left of nbndry have size of ts_left
                tlo = lo + tileidx * ts_left;
                thi = tlo + ts_left - 1;
            } else {
                tileidx = nleft + (ii-nbndry) / ts_right;  // tiles on the right: ts_right
                tlo = lo + tileidx * ts_right + nleft;
                thi = tlo + ts_right - 1;
            }
        };
        const IntVect& small = box.smallEnd();
        const IntVect& big   = box.bigEnd();
        IntVect ntiles, ivIndex, tilelo, tilehi;

        AMREX_D_TERM(int iv0 = amrex::min(amrex::max(iv[0], small[0]), big[0]);,
		     int iv1 = amrex::min(amrex::max(iv[1], small[1]), big[1]);,
		     int iv2 = amrex::min(amrex::max(iv[2], small[2]), big[2]););

        AMREX_D_TERM(tiling_1d(iv0, small[0], big[0], bin_size[0], ntiles[0], ivIndex[0], tilelo[0], tilehi[0]);,
		     tiling_1d(iv1, small[1], big[1], bin_size[1], ntiles[1], ivIndex[1], tilelo[1], tilehi[1]);,
		     tiling_1d(iv2, small[2], big[2], bin_size[2], ntiles[2], ivIndex[2], tilelo[2], tilehi[2]););

        return AMREX_D_TERM(ivIndex[0], + ntiles[0]*ivIndex[1], + ntiles[0]*ntiles[1]*ivIndex[2]);
	}
    };

    struct assignParticleGridAndTile
    {        
        GeometryData domain;
        Box box;
        BaseFab<int>* mask_ptr;
        
        AMREX_GPU_HOST_DEVICE
        assignParticleGridAndTile(const GeometryData& a_domain,
				  const Box&          a_box,
				  BaseFab<int>*       a_mask_ptr) 
            : domain(a_domain), box(a_box), mask_ptr(a_mask_ptr) {}
        
        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
        thrust::tuple<int, int> operator()(const Particle& p) const {
	    IntVect iv = IntVect(
	AMREX_D_DECL(floor((p.m_rdata.pos[0]-domain.ProbLo()[0])/domain.CellSize()[0]),
		     floor((p.m_rdata.pos[1]-domain.ProbLo()[1])/domain.CellSize()[1]),
		     floor((p.m_rdata.pos[2]-domain.ProbLo()[2])/domain.CellSize()[2]))
				 );

	    iv += domain.Domain().smallEnd();

	    return thrust::make_tuple((*mask_ptr)(iv, 0), (*mask_ptr)(iv, 1));
        }
    };

    struct assignParticleCell
    {        
        GeometryData domain;
        
        AMREX_GPU_HOST_DEVICE
        assignParticleCell(const GeometryData& a_domain)
            : domain(a_domain) {}
        
        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
        IntVect operator()(const Particle& p) const {
	    IntVect iv = IntVect(
	AMREX_D_DECL(floor((p.m_rdata.pos[0]-domain.ProbLo()[0])/domain.CellSize()[0]),
		     floor((p.m_rdata.pos[1]-domain.ProbLo()[1])/domain.CellSize()[1]),
		     floor((p.m_rdata.pos[2]-domain.ProbLo()[2])/domain.CellSize()[2]))
				 );

	    iv += domain.Domain().smallEnd();

	    return iv;
        }
    };
    
    struct grid_is
    {
        int grid_id;
        
        AMREX_GPU_HOST_DEVICE
        grid_is(int a_grid_id) : grid_id(a_grid_id) {}
        
        AMREX_GPU_HOST_DEVICE
        bool operator()(int grid) const
        {
            return grid_id == grid;
        }
    };
    
    struct grid_is_not
    {
        int grid_id;
        
        AMREX_GPU_HOST_DEVICE
        grid_is_not(int a_grid_id) : grid_id(a_grid_id) {}
    
        AMREX_GPU_HOST_DEVICE
        bool operator()(int grid) const
        {
            return grid_id != grid;
        }
    };

    struct getPosition
    {        
        AMREX_GPU_HOST_DEVICE
        getPosition() {}
        
        template <typename Particle>
        AMREX_GPU_HOST_DEVICE
        thrust::tuple<AMREX_D_DECL(Real, Real, Real)> operator()(const Particle& p) const
        {
            return thrust::make_tuple(AMREX_D_DECL(p.m_rdata.pos[0],
                                                   p.m_rdata.pos[1], 
                                                   p.m_rdata.pos[2]));
        }
    };

    template <typename Particle>
    struct setPosition
    {
        AMREX_GPU_HOST_DEVICE
        setPosition() {}

        template <typename Tuple>
        AMREX_GPU_HOST_DEVICE
        Particle operator()(const Tuple& pos) const
        {
            Particle p;
            p.m_rdata.pos[0] = thrust::get<0>(pos);
#if AMREX_SPACEDIM > 1
            p.m_rdata.pos[1] = thrust::get<1>(pos);
#endif
#if AMREX_SPACEDIM > 2
            p.m_rdata.pos[2] = thrust::get<2>(pos);
#endif
            return p;
        }
    };
}

}

#endif 

#endif
