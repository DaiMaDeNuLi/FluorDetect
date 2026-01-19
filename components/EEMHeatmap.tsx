import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { EEMPoint } from '../types';

interface Props {
  data: EEMPoint[];
  width?: number;
  height?: number;
}

export const EEMHeatmap: React.FC<Props> = ({ data, width = 400, height = 300 }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 60, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // X Axis: Emission (发射波长)
    // 根据用户要求，确保 X 轴为 Emission (Col 2)
    const xDomain = d3.extent(data, d => d.em) as [number, number];
    const xScale = d3.scaleLinear()
      .domain(xDomain)
      .range([0, innerWidth]);

    // Y Axis: Excitation (激发波长)
    // 根据用户要求，确保 Y 轴为 Excitation (Col 1)
    const yDomain = d3.extent(data, d => d.ex) as [number, number];
    const yScale = d3.scaleLinear()
      .domain(yDomain)
      .range([innerHeight, 0]); // Cartesian: 0 at bottom

    // Color Scale: Intensity
    const maxIntensity = d3.max(data, d => d.intensity) || 100;
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, maxIntensity]);

    // Determine cell size
    // Calculate precise cell dimensions assuming grid data
    const uniqueEx = Array.from(new Set(data.map(d => d.ex))).sort((a,b)=>a-b);
    const uniqueEm = Array.from(new Set(data.map(d => d.em))).sort((a,b)=>a-b);
    
    // Estimate step size if possible, otherwise average
    const exStep = uniqueEx.length > 1 ? (uniqueEx[uniqueEx.length-1]! - uniqueEx[0]!) / (uniqueEx.length - 1) : 1;
    const emStep = uniqueEm.length > 1 ? (uniqueEm[uniqueEm.length-1]! - uniqueEm[0]!) / (uniqueEm.length - 1) : 1;
    
    const cellWidth = innerWidth / (uniqueEm.length || 1);
    const cellHeight = innerHeight / (uniqueEx.length || 1);

    // Draw Heatmap Rects
    // Note: yScale(d.ex) gives the pixel position of the value. 
    // Since we want the value to represent the "bottom" or "center" of the bin in Cartesian, 
    // and SVG draws rects downwards:
    // We position y at yScale(d.ex) - cellHeight to draw the box "sitting" on the value (if value is top of bin)
    // or typically we align it so the tick is centered. 
    // For simplicity here, we draw 'up' from the coordinate by subtracting height in SVG space.
    g.selectAll("rect")
      .data(data)
      .enter()
      .append("rect")
      .attr("x", d => xScale(d.em))
      .attr("y", d => yScale(d.ex) - cellHeight) 
      .attr("width", cellWidth + 0.5) 
      .attr("height", cellHeight + 0.5)
      .attr("fill", d => colorScale(d.intensity));

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 35)
      .attr("fill", "#334155") // slate-700
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text("发射波长 (Emission) / nm"); // 中文标签

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -innerHeight / 2)
      .attr("fill", "#334155")
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text("激发波长 (Excitation) / nm"); // 中文标签
      
    // Color Legend
    const legendHeight = innerHeight;
    const legendWidth = 15;
    const legendScale = d3.scaleLinear().domain([0, maxIntensity]).range([legendHeight, 0]);
    
    const legendAxis = d3.axisRight(legendScale).ticks(5);
    
    const legendG = svg.append("g")
        .attr("transform", `translate(${width - 45}, ${margin.top})`);
        
    const defs = svg.append("defs");
    const linearGradient = defs.append("linearGradient")
        .attr("id", "linear-gradient")
        .attr("x1", "0%")
        .attr("y1", "100%")
        .attr("x2", "0%")
        .attr("y2", "0%");
    
    const ticks = d3.ticks(0, maxIntensity, 10);
    const gradientData = ticks.map((t, i) => ({
      offset: `${100 * i / (ticks.length - 1)}%`,
      color: colorScale(t)
    }));

    linearGradient.selectAll("stop")
        .data(gradientData)
        .enter().append("stop")
        .attr("offset", (d: any) => d.offset)
        .attr("stop-color", (d: any) => d.color);
        
    legendG.append("rect")
        .attr("width", legendWidth)
        .attr("height", legendHeight)
        .style("fill", "url(#linear-gradient)");
        
    legendG.append("g")
        .attr("transform", `translate(${legendWidth}, 0)`)
        .call(legendAxis);

  }, [data, width, height]);

  return <svg ref={svgRef} width={width} height={height} className="bg-white rounded shadow-sm border border-slate-200" />;
};