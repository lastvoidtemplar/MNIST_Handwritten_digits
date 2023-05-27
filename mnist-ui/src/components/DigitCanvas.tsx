import { useRef, useState } from 'react';
import { Point, useOnDraw } from '../hooks/useOnDraw';
import { invoke } from '@tauri-apps/api/tauri'

interface Size{
  width:number,
  height:number
}

function DigitCanvas({
  width,
  height
}:Size) {
  const {
    setCanvasRef,
    onCanvasMouseDown,
    clearCanvas,
    getImage
} = useOnDraw(onDraw);


const [digit, setDigit] = useState(0)
function onDraw(ctx:CanvasRenderingContext2D, point:Point, prevPoint:Point) {
    drawLine(prevPoint, point, ctx, '#ffffff', 20);
}

function drawLine(
    start:Point,
    end:Point,
    ctx:CanvasRenderingContext2D,
    color:string,
    width:number
) {
  if(end){
    start = start ?? end;
    ctx.beginPath();
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.lineCap = 'round' 
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  }
    

}

return(
  <>
    <canvas
        width={width}
        height={height}
        onMouseDown={onCanvasMouseDown}
        className='border-black border-4 border-solid bg-black'
        ref={setCanvasRef}
    />
    <button onClick={clearCanvas}>Clear</button>
    <button onClick={()=>{
      const image = getImage()
      console.log(image);
      
      invoke('preidict',{image}).then((res) => setDigit(res as number) )
    } 
    }>Pridict</button>
    <label >{digit}</label>
</>
);

}

export default DigitCanvas